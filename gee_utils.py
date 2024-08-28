import ee

# general
import numpy as np
from itertools import product

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
    """Display key metrics from tif object"""
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
    # I think b1 = algae, b2 = sand, b3 = coral
    bands = {}
    for band in tif_object.indexes:
        b = tif_object.read(band)
        b[b == -9999] = np.nan
        bands[f"b{band}"] = b
    
    return bands


def get_value_distribution_from_image(band, aoi, scale: int=8, maxPixels: int=1e9):
    """
    Calculates the value distribution of a given band within a specified area of interest (AOI).
    
    Parameters:
    band (ee.Image): The band for which the value distribution is calculated.
    aoi (ee.Geometry): The area of interest (AOI) within which the value distribution is calculated.
    Returns:
    hist_values (numpy.ndarray): An array containing the histogram values.
    hist_counts (numpy.ndarray): An array containing the bucket means.
    Note:
    The scale parameter should be adjusted according to the specific dataset.
    The maxPixels parameter specifies the maximum number of pixels to include in the computation.
    """
    
    hist_dict = band.reduceRegion(
        reducer=ee.Reducer.histogram(),
        geometry=aoi,
        scale=scale,  # Adjust the scale to your specific dataset
        maxPixels=maxPixels
    ).getInfo()

    hist_values = np.array(hist_dict[band.bandNames().getInfo()[0]]['bucketMeans'])
    hist_counts = np.array(hist_dict[band.bandNames().getInfo()[0]]['histogram'])
    return hist_values, hist_counts


def alpha_band(image, band_name: str):
    """
    Apply opacity mask to layer (zero values = transparent)
    """
    alpha = image.select(band_name)
    return image.select(band_name).updateMask(alpha)


def legend_gradient(map, title, visual, position='bottom-right'):
    # Create the map object

    # Create the legend panel
    legend = geemap.map().add_legend(title=title, title_font_size='15px', 
                                     position=position, width='100px')

    # Create the gradient image
    lon = ee.Image.pixelLonLat().select('latitude')
    gradient = lon.multiply((visual['max'] - visual['min']) / 100.0).add(visual['min'])
    legend_image = gradient.visualize(**visual)
    
    # Generate a thumbnail from the image
    thumbnail = geemap.ee_to_thumbnail(legend_image, dimensions='10x50', bbox='0,0,10,100')

    # Add the title, gradient image, and labels to the legend panel
    legend.add_child(geemap.Label(title, fontSize='15px', fontWeight='bold', textAlign='center'))
    legend.add_child(thumbnail)
    legend.add_child(geemap.Label(str(visual['max']), textAlign='center'))
    legend.add_child(geemap.Label(str(visual['min']), textAlign='center'))

    # Add the legend to the map
    Map.add(legend)
    return Map


def resample_histogram(central_values, bin_counts, bin_range: tuple[float]=(0,1), nbins: int = 100):
    """Used when central_values are not evenly spaced e.g. non-regularly spaced central values"""
    # Calculate bin edges from central values
    bin_widths = np.diff(central_values)  # Width of each bin
    bin_edges = np.concatenate((
        [central_values[0] - bin_widths[0] / 2],  # Starting edge
        central_values[:-1] + bin_widths / 2,  # Right edges
        [central_values[-1] + bin_widths[-1] / 2]  # Ending edge
    ))

    # Define the new, evenly spaced bin edges
    new_bin_edges = np.linspace(min(bin_range), max(bin_range), nbins+1)  # 10 new bins between 0 and 1

    # Calculate the new bin counts
    new_bin_counts = np.zeros(len(new_bin_edges) - 1)

    # Vectorized calculation of bin overlaps and resampling
    original_bin_ranges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T
    new_bin_ranges = np.vstack([new_bin_edges[:-1], new_bin_edges[1:]]).T

    # Calculate overlap between original bins and new bins
    overlap_matrix = np.maximum(0, np.minimum(original_bin_ranges[:, None, 1], new_bin_ranges[None, :, 1]) -
                                np.maximum(original_bin_ranges[:, None, 0], new_bin_ranges[None, :, 0]))

    # Calculate the fraction of each original bin's count to assign to each new bin
    proportion_matrix = overlap_matrix / (original_bin_ranges[:, 1] - original_bin_ranges[:, 0])[:, None]

    # Multiply the proportions by the original counts and sum over the original bins
    new_bin_counts = np.sum(proportion_matrix * bin_counts[:, None], axis=0)

    return new_bin_edges, new_bin_counts


# This helper function returns a list of new band names.
def get_new_band_names(
    prefix, 
    band_names=ee.List(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])):
  seq = ee.List.sequence(1, band_names.length())

  def add_prefix_and_number(b):
    return ee.String(prefix).cat(ee.Number(b).int())

  return seq.map(add_prefix_and_number)


def get_principal_components(centered, scale, region):
  # Collapse bands into 1D array
  arrays = centered.toArray()

  # Compute the covariance of the bands within the region.
  covar = arrays.reduceRegion(
      reducer=ee.Reducer.centeredCovariance(),
      geometry=region,
      scale=scale,
      maxPixels=1e9,
  )

  # Get the 'array' covariance result and cast to an array.
  # This represents the band-to-band covariance within the region.
  covar_array = ee.Array(covar.get('array'))

  # Perform an eigen analysis and slice apart the values and vectors.
  eigens = covar_array.eigen()

  # This is a P-length vector of Eigenvalues.
  eigen_values = eigens.slice(1, 0, 1)
  # This is a PxP matrix with eigenvectors in rows.
  eigen_vectors = eigens.slice(1, 1)

  # Convert the array image to 2D arrays for matrix computations.
  array_image = arrays.toArray(1)

  # Left multiply the image array by the matrix of eigenvectors.
  principal_components = ee.Image(eigen_vectors).matrixMultiply(array_image)

  # Turn the square roots of the Eigenvalues into a P-band image.
  sd_image = (
      ee.Image(eigen_values.sqrt())
      .arrayProject([0])
      .arrayFlatten([get_new_band_names('sd')])
  )

  # Turn the PCs into a P-band image, normalized by SD.
  return (
      # Throw out an an unneeded dimension, [[]] -> [].
      principal_components.arrayProject([0])
      # Make the one band array image a multi-band image, [] -> image.
      .arrayFlatten([get_new_band_names('pc')])
      # Normalize the PCs by their SDs.
      .divide(sd_image)
  )


def compute_pca(image, roi):
# Perform PCA
    pca = image.reduceRegion(
        reducer=ee.Reducer.principalComponents(3),
        geometry=roi,
        scale=30,
        maxPixels=1e6
    )
    return pca


def get_polygon_centre_from_coordinates(coords: list[tuple]) -> tuple:
    """Returns the centre of a polygon from a list of coordinates"""
    unique_coords = coords[:-1]
    x = [c[0] for c in unique_coords]
    y = [c[1] for c in unique_coords]
    return (sum(x) / len(unique_coords), sum(y) / len(unique_coords))


def tune_n_trees(n_trees, train_set, val_set, image):
    # To start with, twice the number of covariates you have
    classifier = ee.Classifier.smileRandomForest(n_trees)\
        .train(train_set, "class", image.bandNames())
        # need to specify which bands (not interested in random)
    
    accuracy = val_set.classify(classifier)\
        .errorMatrix("class", "classification").accuracy()
    
    return accuracy


def classify_by_max_band(image, band_names):
    """
    Classifies each pixel in the image based on the band with the highest value.

    Args:
    image (ee.Image): The input image with multiple bands.
    band_names (list): A list of band names to compare.

    Returns:
    ee.Image: An image where each pixel is classified by the band with the highest value.
    """

    # Create an expression to find the index of the band with the maximum value
    max_band_index = image.expression(
        " + ".join([f"(band_{i} >= " + f" && band_{i} >= ".join([f"band_{j}" for j in range(len(band_names)) if j != i]) + f") * {i+1}"
                    for i in range(len(band_names))]),
        {f'band_{i}': image.select(band_names[i]) for i in range(len(band_names))}
    )
    # Stack the bands into a single image collection
    band_stack = image.select(band_names).toArray()
    
    # Find the index of the maximum value for each pixel
    max_band_index = band_stack.arrayArgmax().arrayGet([0]).add(1)
    
    # Rename to 'Class' for clarity
    classified_image = max_band_index.rename('class')
    
    return classified_image


def tune_with_params(params, train_set, val_set, image):
    """Trains a classifier with given parameters and evaluates accuracy."""
    
    # Build the classifier with the current set of parameters
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=params.get("numberOfTrees", 200),
        variablesPerSplit=params.get("variablesPerSplit", None),
        minLeafPopulation=params.get("minLeafPopulation", 1),
        bagFraction=params.get("bagFraction", 0.5),
        maxNodes=params.get("maxNodes", None),
        seed=params.get("seed", 42)
    ).train(train_set, "class", image.bandNames())
    
    confusion_matrix = val_set.classify(classifier).errorMatrix("class", "classification")
    # Classify the validation set and calculate accuracy
    accuracy = confusion_matrix.accuracy()
    f1_score = confusion_matrix.fscore(beta=1)
    # TODO: calculate balanced accuracy
    # balanced_accuracy = calc_balanced_accuracy_ee(confusion_matrix.array())
    
    return ee.Dictionary({
        "params": params,
        "accuracy": accuracy,
        "f1_score": f1_score,
        # "balanced_accuracy": balanced_accuracy
    })


def grid_search(params_grid, train_set, val_set, image):
    """
    Performs a grid search over the specified parameters
    """
    # Generate all combinations of parameters
    keys, values = zip(*params_grid.items())
    param_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    
    # Convert the parameter combinations to an Earth Engine list of dictionaries
    param_combinations_ee = ee.List(param_combinations)
    
    # Map the tune_with_params function over the parameter combinations
    results = param_combinations_ee.map(lambda params: tune_with_params(ee.Dictionary(params), train_set, val_set, image))
    
    return results