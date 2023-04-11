from prettytable import PrettyTable

def cm_to_ious(cm):
    mean_ious = []
    for i in range(16):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        mean_ious.append(tp / union)
    
    return mean_ious

def format_results(mean_ious, return_dic=False):
    class_map = {
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values()) + ['mean']
    class_ious = mean_ious + [sum(mean_ious) / len(mean_ious)]
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = round(cls_iou, 3)
        x.add_row([cls_name, round(cls_iou, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
