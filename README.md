## Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download. 
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 75k products with complete product details and prices
- **Test Set:** 75k products for final evaluation


### File Descriptions:

*Source files*

1. **src/utils.py:** Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues.
2. **sample_code.py:** Sample dummy code that can generate an output file in the given format. Usage of this file is optional.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv



