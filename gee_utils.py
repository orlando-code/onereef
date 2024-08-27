import ee

# general
import numpy as np

# plotting
import folium


def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)


# Creating land mask with NDWI and Random Forest
def landmask(img):
    """
    Applies a land mask to the input image based on the normalized difference water index (NDWI).
    Parameters:
    img (ee.Image): The input image to apply the land mask to.
    Returns:
    ee.Image: The input image with the land mask applied.
    """
    
    ndwi = img.normalizedDifference(['B3', 'B8'])
    training = ndwi.sampleRegions(collection=landGeoms.merge(waterGeoms), properties=['class'], scale=3)
    trained = ee.Classifier.smileRandomForest(10).train(features=training, classProperty='class')
    classified = ndwi.classify(trained)
    mask = classified.eq(1)
    return img.updateMask(mask)


def inspect_tif(tif_object) -> None:
    print("DIMENSIONS | height: ", tif_object.height, "width: ", tif_object.width)
    print("BOUNDS | ", tif_object.bounds)
    print("BANDS | count: ", tif_object.count)
    print("DATA TYPE(S) | ", {i: dtype for i, dtype in zip(tif_object.indexes, tif_object.dtypes)})
    print("CRS | ", tif_object.crs)
    print("TRANSFORM | ", tif_object.transform)


def process_tif(tif_object) -> dict:
    """
    Processes a tif object and returns a dictionary of bands. Replaces -9999 with np.nan.
    
    Parameters:
    tif_object (rasterio.io.DatasetReader): The tif object to process.
    Returns:
    dict: A dictionary of bands (label and array).
    """
    # I think b1 = coral, b2 = sand, b3 = algae
    bands = {}
    for band in tif_object.indexes:
        b = tif_object.read(band)
        b[b == -9999] = np.nan
        bands[f"b{band}"] = b
    
    return bands