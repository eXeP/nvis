import torch

def test(h=1024, w=1024):
    test_image = torch.randn((1, 3, h, w))
    test_image2 = torch.randn((2, 3, h, w))
    from IPython import embed; embed()
    # %run /home/exep/research/nvis/vis.py
    # vis(test_image, test_image2)
    # test_image3 = torch.randn((1, 3, h, w))
    # vis(test_image3, append=True)

if __name__ == '__main__':
    test()