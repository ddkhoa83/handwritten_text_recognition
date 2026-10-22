
#Import Path to build and manage filesystem paths
from pathlib import Path
#Import struct to decode binary headers into Python integers.
import struct
#Import pyplot to display MNIST images
import numpy as np
#Import NDArray to describe NumPy array types in type annotations.
from numpy.typing import NDArray
#Import the function that splits data into training and validation subsets
from sklearn.model_selection import train_test_split

# Tự động lấy thư mục hiện tại chứa file script này (chính là thư mục data)
data_dir: Path = Path(__file__).parent / "MNIST" / "raw"


#Define a function that receives the path to an extracted MNIST image file.
def load_mnist_images(file_path: str | Path) -> NDArray[np.uint8]:
    """Load MNIST images from an uncompressed IDX file.
    Parameters
    file_path: str|Path
        Location of the extracted image file, without the .gz extension.
    Returns
    NDArray[np.uint8]
        Pixel intensities with shape (n_images, n_rows, n_cols).
    """
    #Open the file in binary read mode:
    #"r" means read, and "b" means binary.
    #"with" automatically close the file when this block ends,
    #including when a exception occurs.
    with open(file_path, "rb") as file:

        #Read the first 16 bytes, which contain the four header integers.
        #After this operation, the file position is immediately after the header.
        header = file.read(16)

        #Check that the complete header was read.
        #A shorter header means that the file is incomplete or invalid.
        if len(header) != 16:
            #Stop the function and explain the problem
            raise ValueError(f"Incomplete image header: {file_path}")

        #The magic number identifies the file format
        #Decode the header into four integers.
        #">" specifies big-endian byte order.
        #Each "I" specifies an unsigned 4-byte integer
        #"IIII" therefore decodes four integers from 16 bytes.
        #Assign the decoded integers to the four variables in order.
        magic: int
        n_images: int
        n_rows: int
        n_cols: int
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", header)
        #MNIST image files must have the magic number 2051
        if magic != 2051:
            #Reject a file that does not have the expected image-file format
            raise ValueError(f"Invalid image file: {file_path}")
        
        #Check that the image count and both image dimensions are positive
        if n_images <= 0 or n_rows <= 0 or n_cols <= 0:
            #Stop if the header contains invalid dimension.
            raise ValueEror(f"Invalid image dimension: {file_path}")

        #Read everything remaining after the header.
        #Each remaining byte represents one pixel intensity
        pixel_bytes: bytes = file.read()

        #Interpret the raw bytes as a one-dimensional Numpy array.
        #dtype=np.uint8 makes each byte one integer in [0, 255]
        #frombuffer interprets existing bytes rather than parsing text
        pixels: NDArray[np.uint8] = np.frombuffer(
            pixel_bytes,
            dtype = np.uint8,
        )

    #Compute the number of pixels expected from the header.
    #For training data: 60000*28*28 = 47.040.000.
    expected_size: int = n_images*n_rows*n_cols
    #Pixels.size is the total number of elements actually read
    #Compare it with the expected number of elements.
    if pixels.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} pixels, found {pixels.size}."
        )

    #Reshape the flat array into a collection of two-dimensional images.
    #Dimension 0 identifies the image
    #Dimension 1 identifies the row within that image
    #Dimension 2 identifies the column and the uint8 element type.
    #This preserves both pixel order and the uint8 element type.
    images: NDArray[np.uint8] = pixels.reshape(
        n_images,
        n_rows,
        n_cols,
    )
    #Send the resulting image array back to the caller
    return images

#Define a function accepting either a string or a Path
#It returns a Numpy array containing signed 64-bit integers.
def load_mnist_labels(file_path: str | Path) -> NDArray[np.int64]:
    """
    Load an uncompressed MNIST IDX label file.

    Parameters
    file_path: str | Path
        Location of the extracted label file.
    Returns
    NDArray[np.float64]
        Digit labels from 0 to 9, with shape (n_labels, ).
    """
    #Open the file in binary read mode.
    #Automatically close it when excution leaves the with block
    with open(file_path, "rb") as file:
        #Read the 8-byte header
        #It contains the magic number and the number of labels.
        header: bytes = file.read(8)

        #Verify that the complete header is present.
        if len(header) != 8:
            #Stop if the header is shorter than required.
            raise ValueError(f"Incomplete label header: {file_path}")

        #Decode two unsigned 4-byte integers using big-endian order.
        #Assign the first to magic and the second to n_labels
        magic: int
        n_labels: int 
        magic, n_labels = struct.unpack(">II", header)

        #MNIST label files must have the magic number 2049.
        if magic != 2049:
            #Reject files with an unexpected format identifier.
            raise ValueError(f"Invalid label file: {file_path}")

        #Read the remaining bytes and interpret each byte as one label.
        #The initial label representation in unit8.
        #The resulting array is one-dimensional.
        raw_labels: NDArray[np.uint8] = np.frombuffer(
            file.read(),
            dtype = np.uint8,
        )

        #Verify that the actual number of labels matches the header.
        if raw_labels.size != n_labels:
            #Report the expected an actual label counts.
            raise ValueError(
                f"Expected {n_labels} labels, found {raw_labels.size}"
            )

        #raw_labels > 9 creates a Boolean array.
        #np.any returns True if at least one label is greater than 9.
        #Negative values are impossible here because the dtype is uint8

        if np.any(raw_labels > 9):
            #Reject labels outside the valid digit classes
            raise ValueError("MNIST labels must be between 0 and 9.")

        #Convert the actual array element type from uint8 to int64.
        #The numerical labels remain unchanged.
        #For example, the digit label 7 remain 7.
        labels: NDArray[np.int64] = raw_labels.astype(np.int64)

        #Return the one-dimensional integer label array.
        return labels


#Define a function accepting an array of uint8 image pixels.
#The function returns an array of float32 features.

def preprocess_images(
    images: NDArray[np.uint8],
) -> NDArray[np.float32]:
    """
    Flatten MNIST images and scale their pixel values to [0, 1].

    Input:
    Shape: (n_images, 28, 28)
    Element type: unit8.
    Ouput:
    Shape: (n_images, 784).
    Element type: float32
    """
    #Images.ndim gives the number of array dimensions.
    #We require three dimensions: image, row, and column.
    #images.shape[1:] gives all dimension after the first

    if images.ndim != 3 or images.shape[1:] != (28, 28):
        #Stop if the input does not have the expected MNIST image shape.
        raise ValueError(
            "Expected images with shape (n_images, 28, 28)."
        )

    #Read the first dimension, which is the number of images.
    n_images: int = images.shape[0]

    #Flatten each 28*28 image into a vector of 784 pixels.
    #Each row now represent one complete image
    #Each column represents a fixed pixel position.
    #Flattening does not change the pixel values of their dtype.
    flat_images: NDArray[np.uint8] = images.reshape(n_images, 784)

    #Convert pixel values into 32-bit floating-point numbers.
    #This allows normalized fractional values such as 0.5.
    #A float32 element occupies 4 bytes.

    float_images: NDArray[np.float32] = flat_images.astype(np.float32)

    #Divide everay pixel by 255.
    #np.float32(255.0) explicitly creates a float32 divisor.
    #Original values in [0, 255] become values in [0, 1].

    scaled_images: NDArray[np.float32] = (
        float_images / np.float32(255.0)
    )

    #Return the flattened and normalized features.
    return scaled_images

#Build the training-image path and pass it to the image loader.
#Expected shape: (60000, 28, 28)
train_images: NDArray[np.uint8] = load_mnist_images(
    data_dir / "train-images-idx3-ubyte"
)

#Load the origin test images.
#Expected shape: (10000, 28, 28)
test_images: NDArray[np.uint8] = load_mnist_images(
    data_dir / "t10k-images-idx3-ubyte"
)

#Load one label for each training image.
#Expected shape: (60000,).
train_labels: NDArray[np.int64] = load_mnist_labels(
    data_dir / "train-labels-idx1-ubyte"
)

#Load one label for each test image.
#Expected shape: (10000,).
test_labels: NDArray[np.int64] = load_mnist_labels(
    data_dir / "t10k-labels-idx1-ubyte"
)

#len() returns the size of the first dimension of three arrays.
#Check that the training image and label counts agree.
if len(train_images) != len(train_labels):
    #Stop if some training images do not have corresponding labels.
    raise ValueError("Training image and label counts do not match.")

#Check that the test image and label counts agree.
if len(test_images) != len(test_labels):
    #Stop if the test counts are inconsistent
    raise ValueError("Test image and label counts do not match.")

#Flatten and normalize all 60000 original training images.
#Expected shape: (60000, 784)
#Expected dtype: float32
X_train_full: NDArray[np.float32] = preprocess_images(train_images)

#Apply the identical preprocessing to the original test images.
#Expected shape: (10000, 784).
#Expected dtype: float32.
X_test_scaled: NDArray[np.float32] = preprocess_images(test_images)

#Give the training labels the variable name used in the ML workflow.
#This assignment references the same array, it does not copy it.
#Labels stay as integers from 0 to 9.
y_train_full: NDArray[np.int64] = train_labels

#Assign the original test labels to y_test.
#Do not normalize class labels.
y_test: NDArray[np.int64] = test_labels

#Declare the type of the training feature array.
#This declaration alone does not assign a value or allocate an array
X_train_scaled: NDArray[np.float32]

#Declare the type of the validation feature array
X_val_scaled: NDArray[np.float32]

#Declare the type of the training label array.
y_train: NDArray[np.int64]

#Declare the type of the validation label array.
y_val = NDArray[np.int64]

#Split features and labels together to preserve their correspondence.
#Unpack the returned arrays in this order:
#training features, validation features, training labels, validation labels.
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    #Supply all 60000 preprocessed training images.
    X_train_full,
    #Supply the labels corresponding to those images.
    y_train_full,

    #Reverse exactly 10000 samples for validation.
    #An integer specifies a sample count, not a proportion
    test_size = 10000,

    #Set the random seed so the same inputs produce a reproducible split.
    random_state = 42,

    #Keep approximately the same class proportions in both subsets.
    #The class information comes from y_train_full.
    stratify = y_train_full,
)

np.savez(
    Path(__file__).parent / 'processed_mnist.npz',
    X_train=X_train_scaled,
    y_train=y_train,
    X_val=X_val_scaled,
    y_val=y_val,
    X_test=X_test_scaled,
    y_test=y_test,
)
print('Đã lưu dữ liệu tiền xử lý thành công vào processed_mnist.npz!')