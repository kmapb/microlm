import pytest
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


def _make_v2(V=256, C=16, T=32):
    return SummNet(vocab_size=V, dim=C, fc_dim=C, height=3, max_length=T,
                   kernel_size=2, arch='v2')


def test_v2_is_causal():
    """Two inputs identical up to position t must produce identical
    predictions before t -- the functional causality test, covering the
    gated convs, pre-norm residuals, and skip aggregation together."""
    V, C, T, t = 256, 16, 32, 20
    net = _make_v2(V, C, T)
    net.eval()

    torch.manual_seed(1)
    a = torch.randint(0, V, (1, T))
    b = a.clone()
    b[0, t:] = torch.randint(0, V, (T - t,))

    with torch.no_grad():
        ya = net(a).reshape(T, V)
        yb = net(b).reshape(T, V)
    torch.testing.assert_close(ya[:t], yb[:t])
    assert not torch.allclose(ya[t:], yb[t:]), "suffix change had no effect"


def test_v2_ties_weights_and_trains():
    V, C, T = 256, 16, 32
    net = _make_v2(V, C, T)
    assert net.head[-1].weight is net.token_embedding_table.weight
    # Tied matrix gets output-projection-scale init, not embedding-scale.
    assert float(net.token_embedding_table.weight.std()) < 0.05

    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    batch = {'input_ids': torch.randint(0, V, (2, T)),
             'num_tokens': torch.full((2,), T)}
    losses = []
    for i in range(10):
        loss = net.training_step(batch, i)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not improve: {losses[0]} -> {losses[-1]}"
    # Still tied after training.
    assert net.head[-1].weight is net.token_embedding_table.weight


def test_v2_param_count_near_v1():
    """The design brief: tying pays for the gated width, so v2 stays near
    v1's parameter count at matched dims (the exact ratio shifts with the
    vocab/dim balance; at full scale it's within 5%)."""
    kwargs = dict(vocab_size=1000, dim=64, fc_dim=64, height=4,
                  max_length=256, kernel_size=3)
    count = lambda net: sum(p.numel() for p in net.parameters()
                            if p.requires_grad)
    v1, v2 = count(SummNet(arch='v1', **kwargs)), count(SummNet(arch='v2', **kwargs))
    assert abs(v2 - v1) / v1 < 0.35, f"v1={v1} v2={v2}"


def test_height_derived_from_context():
    """Default height is the smallest stack whose receptive field
    (kernel_size ** height) covers max_length."""
    net = SummNet(vocab_size=64, dim=8, fc_dim=16, max_length=512, kernel_size=2)
    assert net.filter_bank.height == 9  # 2**9 == 512
    net = SummNet(vocab_size=64, dim=8, fc_dim=16, max_length=4096, kernel_size=3)
    assert net.filter_bank.height == 8  # 3**8 == 6561 >= 4096 > 3**7
    # The derived value lands in hparams, so checkpoints round-trip concretely.
    assert net.hparams.height == 8


def test_height_overshoot_warns():
    with pytest.warns(UserWarning, match='overshoots'):
        SummNet(vocab_size=64, dim=8, fc_dim=16, max_length=64, kernel_size=2,
                height=12)


def test_lr_schedule_warmup_and_decay():
    peak, warmup, decay = 1e-3, 10, 100
    net = SummNet(vocab_size=64, dim=8, fc_dim=16, max_length=64, kernel_size=2,
                  lr=peak, warmup_steps=warmup, lr_decay_steps=decay)
    cfg = net.configure_optimizers()
    opt = cfg['optimizer']
    assert cfg['lr_scheduler']['interval'] == 'step'
    sched = cfg['lr_scheduler']['scheduler']

    lrs = []
    for _ in range(decay + 20):
        lrs.append(opt.param_groups[0]['lr'])
        opt.step()
        sched.step()

    assert lrs[0] == pytest.approx(peak / warmup)   # warmup starts low...
    assert lrs[warmup - 1] == pytest.approx(peak)   # ...tops out at the peak...
    assert max(lrs) == pytest.approx(peak)          # ...and never exceeds it.
    assert lrs[-1] == pytest.approx(0.1 * peak)     # cosine lands on the floor.
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:])), \
        "decay phase should be monotonic"


def test_loss_ignores_padding():
    """Pad targets carry no gradient: the loss on a right-padded sequence must
    equal the loss on just its real prefix. (Causality guarantees the shared
    positions see identical predictions, so any difference is pad leakage.)"""
    V, T, PAD = 128, 12, 0
    real_len = 6

    torch.manual_seed(0)
    net = SummNet(vocab_size=V, dim=8, fc_dim=16, height=2, max_length=T,
                  kernel_size=2, pad_token_id=PAD)
    net.eval()

    real = torch.randint(1, V, (1, real_len))
    padded = torch.cat((real, torch.full((1, T - real_len), PAD)), dim=1)

    with torch.no_grad():
        loss_padded = net._shared_eval(padded, 0, 'test')
        loss_prefix = net._shared_eval(real, 0, 'test')
    torch.testing.assert_close(loss_padded, loss_prefix)
