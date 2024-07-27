class Config:
    device = 0
    imgsz = [640, 640]
    half = False
    auto = True
    agnostic_nms = False
    classes = None
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 5
    line_thickness = 5
    warmshape_for_object_detection = (1, 3, 640, 640)
    warmshape_for_action_recognition = (1, 3, 60, 112, 112)