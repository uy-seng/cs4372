dataset:
    https://www.kaggle.com/competitions/cifar-10/overview
    
pre-trained model:
    https://www.kaggle.com/models/metaresearch/llama-3.2-vision

use transfer learning

fine tune the pre-built model to suit your dataset

use any one of the deep learning techniques that we studied in class

divide data into training and test data

tune as many parameter as possible
    can refer to documentation and sample code available on tensorflow
    
    keep log of your experiements with the parameters used and accuracy and loss obtained

deliverables:
    history plot
        show the training and testing accuracy and loss as a function of number of iterations.  
    
    example of at least 25 data points from the test dataset, showing the following
        data
        true label
        predicted label

    a table containing the details of parameter testing and tuning. e.g
    
    -------------------------------------------------------------------------------------
    iterations           parameters                         training adn testing accuracy
    -------------------------------------------------------------------------------------
    1.                    number of layers = ...            train = 80% and test = 78%
                          filter size layer 1 = ...
                          activation function = ...
                          .
                          .
                          .
    
    if any assumption is made, state them completely.