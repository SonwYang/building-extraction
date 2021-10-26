from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, opening
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask, d4_image2mask
from skimage.morphology import opening, closing, square
import numpy as np
from osgeo import gdal, ogr

###mask1   big    mask2 small
def my_watershed(mask1, mask2):
    """
    watershed from mask1 with markers from mask2
    """
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(mask1, markers, mask=mask1, watershed_line=True)
    return labels

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj, im_geotrans, im_width, im_height, im_data

if __name__ == '__main__':
    binary_map_path = r'D:\MyWorkSpace\paper\fishpond\fishpond_prediction\predict_seg\test.tif'
    dn_path = r'D:\MyWorkSpace\paper\fishpond\fishpond_prediction\predict_seg\test_dn.tif'
    outPath = r'D:\MyWorkSpace\paper\fishpond\fishpond_prediction\predict_seg\test_watershed.tif'
    threshold = 185

    im_proj, im_geotrans, im_width, im_height, binary = read_img(binary_map_path)
    im_proj, im_geotrans, im_width, im_height, dn = read_img(dn_path)

    binary = np.where(binary == 1, 1, 0)
    mark = np.where(dn > threshold, 1, 0)
    result = my_watershed(binary, mark)
    result[result > 0] = 1
    # result = bool(result)
    result = remove_small_objects(result, 100)
    result = result.astype(np.uint8)
    gdalTools.write_img(outPath, im_proj, im_geotrans, result)
