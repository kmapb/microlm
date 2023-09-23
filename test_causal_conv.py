from summ_net import DilationNet, SummNet
import sys
import torch

def make_causal_conv1d(kernel_size, in_channels, out_channels, dilation_rate):
    from summ_net import CausalConv1d
    return CausalConv1d(kernel_size, in_channels, out_channels, dilation_rate)

def init_c1_params(c1, wval=1.0, bval=0.0):
    c1.state_dict()['conv1d.weight'].fill_(wval)
    c1.state_dict()['conv1d.bias'].fill_(bval)
    
def test_causal_conv_basic():
    B = 1
    C = 4
    T = 100
    filter_width = 4
    
    # No dilation: just channels in, stride of 1
    c1 = make_causal_conv1d(filter_width, C, C, 1).cuda()
    init_c1_params(c1)
    x = torch.zeros(B, C, T).cuda()
    impulse_start_time = 10
    x[0][0][impulse_start_time] = 1.0
    y = c1(x)
    print("Checking stuff!")
    assert y.shape == x.shape
    # None of the signal from impulse_start_time "leaks" into the past
    assert y[:, :, 0:impulse_start_time].sum() == 0.0
    # Signal from impulse_start_time is on in the filter's output
    assert y[:, :, impulse_start_time].sum() == 1.0 * C
    assert y[:, :, impulse_start_time:impulse_start_time+filter_width].sum() == 1.0 * C * filter_width
    # ...and it's back off in the rest of the output
    assert y[: :, impulse_start_time+filter_width:].sum() == 0.0
    print("Checked")
    
    # Backward?

    y_hat = torch.zeros(y.shape).cuda()
    optimizer = torch.optim.SGD(
        c1.parameters(), lr=0.01, momentum=0.9
    )

    for i in range(0, 1000):
        y = c1(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_causal_conv_dilatory(dilation=2):
    B = 1
    C = 4
    T = 100
    filter_width = 2
    impulse_start_time = 10
    
    c1 = make_causal_conv1d(filter_width, C, C, dilation).cuda()
    init_c1_params(c1)

    x = torch.zeros(B, C, T).cuda()
    x[0, 0, impulse_start_time] = 1.0
    y = c1(x)
    print("Checks dilatory {}!".format(dilation))
    # print(y)
    assert y.shape == x.shape
    # Signal doesn't leak into past
    assert y[:, :, 0:impulse_start_time].sum() == 0.0
    # Signal appears
    assert y[:, :, impulse_start_time].sum() == 1.0 * C
    # Signal skips next D - 1 items
    for d in range(1, dilation-1):
        assert y[:, :, impulse_start_time+d].sum() == 0.0
    # Signal reappears at start + D
    assert y[:, :, impulse_start_time+dilation].sum() == 1.0 * C
    # Then disappears again
    assert y[:, :, impulse_start_time+dilation + 1:].sum() == 0.0

    print("Done!")

def test_dilation_net(height=1):
    B, C, T = 1, 3, 20
    
    net = DilationNet(C, height).cuda()
    for c in net.convs():
        init_c1_params(c)
        
    x = torch.zeros(B, C, T).cuda()
    impulse_start_time = 10
    x[0, 0, impulse_start_time] = 1.0
    
    y = net(x)
    print("Testing dilation net height {}".format(height))
    assert y.shape == x.shape
    # Harder to generalize, but still must be the case that no signal leaks into the past...
    assert y[:, :, 0:impulse_start_time].sum() == 0.0
    # And that there is signal once the impulse starts
    assert y[:, :, impulse_start_time].sum() > 0.0

    # Exercise backward, updates weights    
    target = torch.zeros(y.shape).cuda()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=1e-6, momentum=0.9
    )
    for i in range(100):
        y = net(x)
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_summ_net(height=15):
    B, C, T = 1, 384, 100

    net = SummNet(height=10, dim=C).cuda()
    print("net instantiated!")
    x = torch.randint(0, 29000, (C, T)).cuda()
    print("x ~ {}".format(x.shape))
    y = net(x)
    print("y ~ {}".format(y.shape))

    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.01, momentum=0.9
    )

    btch = torch.randint(0, 29000, (B, T)).cuda()
    for i in range(1000):
        b = {'input_ids': btch}
        loss = net.training_step(b, 0)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    test_causal_conv_basic()
    test_causal_conv_dilatory(2)
    test_causal_conv_dilatory(4)
    for h in range(1, 10):
        pass
        # test_dilation_net(h)
    # test_summ_net()

