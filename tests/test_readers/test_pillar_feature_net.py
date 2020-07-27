from det3d.models import builder
import torch
import pdb

def test_pillar_feature_net():
    pillar_feature_net_cfg = dict(
        type="PillarFeatureNet",
        num_filters=[64],
        num_input_features=5,
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=None,
    )

    pillar_feature_net = builder.build_reader(pillar_feature_net_cfg)
    features = torch.rand([83494, 20, 5])
    num_voxels = torch.randint(0, 100, [83494])
    coors = torch.rand([83494, 4])

    features = pillar_feature_net(features, num_voxels, coors)
    pdb.set_trace()
    assert features.shape == torch.Size([83494, 64])