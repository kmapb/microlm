import torch

from summ_net import CausalConv1d, DilationNet, SummNet


def make_causal_conv1d(kernel_size, in_channels, out_channels, dilation_rate):
    return CausalConv1d(kernel_size, in_channels, out_channels, dilation_rate)


def init_c1_params(c1, wval=1.0, bval=0.0):
    c1.state_dict()['conv1d.weight'].fill_(wval)
    c1.state_dict()['conv1d.bias'].fill_(bval)


def test_causal_conv_basic():
    B = 1
    C = 4
    T = 14
    filter_width = 4

    # No dilation: just channels in, stride of 1
    c1 = make_causal_conv1d(filter_width, C, C, 1)
    init_c1_params(c1)
    x = torch.zeros(B, C, T)
    impulse_start_time = 10
    x[0][0][impulse_start_time] = 1.0
    y = c1(x)
    assert y.shape == x.shape
    # None of the signal from impulse_start_time "leaks" into the past
    assert y[:, :, 0:impulse_start_time].sum() == 0.0
    # Signal from impulse_start_time is on in the filter's output
    assert y[:, :, impulse_start_time].sum() == 1.0 * C
    assert y[:, :, impulse_start_time:impulse_start_time + filter_width].sum() == 1.0 * C * filter_width
    # ...and it's back off in the rest of the output
    assert y[:, :, impulse_start_time + filter_width:].sum() == 0.0


def test_causal_conv_backward():
    """The impulse checks above are static; make sure gradients flow too."""
    B, C, T, filter_width = 1, 4, 14, 4
    c1 = make_causal_conv1d(filter_width, C, C, 1)
    init_c1_params(c1)
    x = torch.zeros(B, C, T)
    x[0][0][10] = 1.0

    before = c1.conv1d.weight.detach().clone()
    y_hat = torch.zeros(c1(x).shape)
    optimizer = torch.optim.SGD(c1.parameters(), lr=0.01, momentum=0.9)
    for _ in range(20):
        loss = torch.nn.functional.mse_loss(c1(x), y_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert not torch.equal(before, c1.conv1d.weight), "weights never moved"


def _check_dilatory(dilation):
    B = 1
    C = 4
    T = 100
    filter_width = 2
    impulse_start_time = 10

    c1 = make_causal_conv1d(filter_width, C, C, dilation)
    init_c1_params(c1)

    x = torch.zeros(B, C, T)
    x[0, 0, impulse_start_time] = 1.0
    y = c1(x)
    assert y.shape == x.shape
    # Signal doesn't leak into past
    assert y[:, :, 0:impulse_start_time].sum() == 0.0
    # Signal appears
    assert y[:, :, impulse_start_time].sum() == 1.0 * C
    # Signal skips next D - 1 items
    for d in range(1, dilation - 1):
        assert y[:, :, impulse_start_time + d].sum() == 0.0
    # Signal reappears at start + D
    assert y[:, :, impulse_start_time + dilation].sum() == 1.0 * C
    # Then disappears again
    assert y[:, :, impulse_start_time + dilation + 1:].sum() == 0.0


def test_causal_conv_dilation_2():
    _check_dilatory(2)


def test_causal_conv_dilation_4():
    _check_dilatory(4)


def test_dilation_net():
    B, C, T = 1, 3, 20
    kernel_size = 2

    for height in range(1, 5):
        net = DilationNet(C, height, kernel_size)
        for residual in net.convs():
            # convs() yields the Residual wrappers; the conv is one level down.
            init_c1_params(residual.submodule)

        x = torch.zeros(B, C, T)
        impulse_start_time = 10
        x[0, 0, impulse_start_time] = 1.0

        y = net(x)
        assert y.shape == x.shape
        # Harder to generalize through the stack, but it must still be the case
        # that no signal leaks into the past...
        assert y[:, :, 0:impulse_start_time].sum() == 0.0, f"leak at height {height}"
        # ...and that there is signal once the impulse starts.
        assert y[:, :, impulse_start_time:].abs().sum() > 0.0, f"silent at height {height}"


def test_dilation_net_backward():
    B, C, T = 1, 3, 20
    net = DilationNet(C, 3, 2)
    x = torch.zeros(B, C, T)
    x[0, 0, 10] = 1.0

    target = torch.zeros(net(x).shape)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for _ in range(20):
        loss = torch.nn.functional.mse_loss(net(x), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_summ_net_trains():
    """A few real optimizer steps on a tiny SummNet: loss should come down."""
    B, C, T, V = 2, 32, 50, 512

    net = SummNet(vocab_size=V, dim=C, fc_dim=64, height=4, max_length=T, kernel_size=2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    batch = {
        'input_ids': torch.randint(0, V, (B, T)),
        'num_tokens': torch.full((B,), T),
    }

    losses = []
    for i in range(10):
        loss = net.training_step(batch, i)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0], f"loss did not improve: {losses[0]} -> {losses[-1]}"
    # It counted every token it saw, once per step.
    assert net.total_train_tokens == B * T * len(losses)
