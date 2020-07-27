from det3d.models import builder


def test_point_pillars_scatter():
    point_pillars_scatter_cfg = dict(type="PointPillarsScatter", ds_factor=1, norm_cfg=None)
    self = builder.build_backbone(point_pillars_scatter_cfg)