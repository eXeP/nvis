import torch

def test(h=1024, w=1024):
    test_image = torch.randn((1, 3, h, w))
    test_image3 = torch.randn((2, 3, h, w))
    test_image2 = torch.randn((3, 3, h, w))
    import sys; sys.path.insert(0, '/home/exep/research/nvis'); from vis import vis; vis(test_image, test_image3, test_image2, bidx=None)
    #from IPython import embed; embed()
    # %run /home/exep/research/nvis/vis.py
    # vis(test_image, test_image2)
    # test_image3 = torch.randn((1, 3, h, w))
    # vis(test_image3, append=True)

if __name__ == '__main__':
    test()