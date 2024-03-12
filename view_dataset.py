import h5py

def print_h5_contents(f, prefix=""):
    """Recursively prints the contents of an HDF5 file
    
    Args:
        f (h5py.File): The HDF5 file object to print the contents of
        prefix (str, optional): The prefix to add to the object names. Defaults to "".
    """
    for key in f.keys():
        obj = f[key]
        obj_name = prefix + "/" + key if prefix else key

        if isinstance(obj, h5py.Dataset):
            print(f"{obj_name}: Dataset, Shape: {obj.shape}, Data type: {obj.dtype}")
            
            try:
                print(f"{obj_name}: Data Sample: {obj[:5]}")
            except Exception:
                print(f"{obj_name}: Unable to display data sample")

        elif isinstance(obj, h5py.Group):
            print(f"{obj_name}: Group")
            print_h5_contents(obj, prefix=obj_name)

def write_h5_contents_to_file(f, file, prefix=""):
    """Recursively writes the contents of an HDF5 file to a file
    
    Args:
        f (h5py.File): The HDF5 file object to write the contents of
        file (file object): The file object to write the contents to
        prefix (str, optional): The prefix to add to the object names. Defaults to "".
    """
    for key in f.keys():
        obj = f[key]
        obj_name = prefix + "/" + key if prefix else key

        if isinstance(obj, h5py.Dataset):
            file.write(f"{obj_name}: Dataset, Shape: {obj.shape}, Data type: {obj.dtype}\n")
            
            try:
                file.write(f"{obj_name}: Data Sample: {obj[:5]}\n")
            except Exception:
                file.write(f"{obj_name}: Unable to display data sample\n")

        elif isinstance(obj, h5py.Group):
            file.write(f"{obj_name}: Group\n")
            write_h5_contents_to_file(obj, file, prefix=obj_name)

def h5_contents_txt(h5_filename, txt_filename):
    """Writes the contents of an HDF5 file to a text file
    
    Args:
        h5_filename (str): The path to the HDF5 file
        txt_filename (str): The path to the text file to write the contents to
    """
    with h5py.File(h5_filename, 'r') as h5_file:
        with open(txt_filename, 'w') as txt_file:
            write_h5_contents_to_file(h5_file, txt_file)

    

if __name__ == "__main__":
    # Example usage
    file_path = "/mnt/g/Github/video_summarizer/datasets/summarizer_dataset_summe_google_pool5.h5"
    f = h5py.File(file_path, "r")
    print_h5_contents(f)
    # h5_contents_txt(file_path, "output.txt")
