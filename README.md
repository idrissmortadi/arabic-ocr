# Arabic-Handwritten-OCR
Cloned from https://github.com/Anna868/Arabic-Handwritten-OCR

- To do:
    - [x] Implement a simple benchmark model with pytorch
    - [x] Improve cropping of images to avoid zooming too much on dots.
    - [x] Center images while cropping
    - [x] Inspect labels
        - It's a complicated problem due to the cursive nature of test.
        - Search for a way to split/segment image to find boundaries of characters.
        - Found "Cursive Overlapped Character Segmentation: An Enhanced Approach" article.
        - Use a different core-zone concept: find the horizontal band (10px high) that has the most pixels.
    - [x] Improve refactoring more
    
