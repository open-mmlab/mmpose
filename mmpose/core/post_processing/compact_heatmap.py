import numpy as np


def compact_heatmap(heatmap, threshold=3e-3):
    """Convert Heatmap into a more compact format. The input heatmap is a numpy
    arrayof the shape H x W. The function will find a tight rectangle that
    contain all positions with score > threshold, then crop the heatmap with
    the tight rectangle. The output heatmap will also be converted to float16.
    To restore the original heatmap, a quadruple (x, y, original_w, original_h)
    of type int16 will also be returned.

    Args:
        heatmap (np.ndarray([H, W])): The input heatmap.
        threshold (float): The heatmap score threshold.

    Returns:
        tuple: New Heatmap and Crop Info.
        - new_heatmap (np.ndarray([h, w])): The cropped heatmap.
        - quadruple (np.ndarray([4])): The crop info.
    """

    h, w = heatmap.shape
    y, x = np.where(heatmap > threshold)

    # In which case not a single element is larger than threshold
    if x.shape[0] == 0:
        new_heatmap = np.zeros([0, 0], dtype=np.float16)
        return new_heatmap, np.array([0, 0, w, h], dtype=np.int16)
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    new_heatmap = heatmap[min_y:max_y, min_x:max_x]
    new_heatmap = new_heatmap.astype(np.float16)

    quadruple = np.array([min_x, min_y, w, h], dtype=np.int16)
    return new_heatmap, quadruple


def compact_heatmaps(heatmaps, threshold=3e-3):
    """Convert Heatmap into a more compact format. The input heatmap is a numpy
    arrayof the shape C x H x W. The function apply compact_heatmap to each
    channel and return a list of results.

    Args:
        heatmaps (np.ndarray([C, H, W])): The input heatmap.
        threshold (float): The heatmap score threshold.

    Returns:
        list[tuple]: New Heatmaps and Crop Infos.
    """
    results = []
    for heatmap in heatmaps:
        results.append(compact_heatmap(heatmap, threshold))
    return results


def recover_compact_heatmap(compact_heatmaps):
    """Convert Compact Heatmap into normal format. The input compact_heatmap is
    a list of tuple, each tuple contains the compact heatmap and its
    corresponding quadruple info (x, y, org_h, org_w) (Note that all
    compact_heatmaps should have the same org_h and org_w). The length of the
    list is #joints. The restores heatmap is of the shape.

    #joints x org_h x org_w.

    Args:
        compact_heatmaps (list[tuple]): The list of compact_heatmap tuples.
            Each tuple contains the compact heatmap and the quadruple info.

    Returns:
        heatmap (np.ndarray([k, h, w])): The restored heatmap.
    """
    org_w = [x[1][2] for x in compact_heatmaps]
    org_h = [x[1][3] for x in compact_heatmaps]
    assert len(set(org_w)) == 1 and len(set(org_h)) == 1
    org_w, org_h = org_w[0], org_h[0]
    heatmap = np.zeros([len(compact_heatmaps), org_h, org_w], dtype=np.float32)
    for i, tup in enumerate(compact_heatmaps):
        x, y = tup[1][:2]
        h, w = tup[0].shape
        compact_heatmap = tup[0].astype(np.float32)
        heatmap[i][y:y + h, x:x + w] = compact_heatmap
    return heatmap
