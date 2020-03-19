import numpy as np

try:  # to maintain backwards compatibility with Python 2.7
    import cPickle as pickle
except:
    import pickle
import glob


class Vessel(object):
    """Create a container object that holds properties. Can be easily saved &
       loaded.
    USAGE
        Create a Vessel instance:
            >>> data = Vessel('storage.dat')
        Assign properties to the object:
            >>> data.variable = [1,2,3,'string data']
            >>> data.my_array = np.arange(0,100,0.1)
            >>> data.my_dict = {'name': 'Albert Einstein'}
        Save the object! If no filename is specified, it will use the initially
        supplied filename, if present; otherwise, an error is thrown.
            >>> data.save()
        When we want to load the data, simply create another instance. A
        filename may be passed during object creation. If the filename
        corresponds to an existing file, the file will automatically be loaded:
            >>> other = Vessel('storage.dat')
        Otherwise, a file may be loaded explicitly, at some later point:
            >>> other.load('other_file.dat')
            >>> other.variable # ==> [1,2,3,'string data']
            >>> other.my_dict  # ==> {'name': 'Albert Einstein'}
            >>> other.my_array # ==> array([ 0. ,  0.1,  0.2,  0.3, ... 9.9])
            >>> other.keys     # ==> ['my_array', 'my_dict', 'variable',
            >>>                #      'current_filename']
        When the .save() method is later called, the current filename will be
        used, unless another filename is explicitly specified as a parameter to
        the save command:
            >>> other.save('new_file.dat') # ==> Saved to a new file!
    TIPS
        To determine the properties attached to an object instance, examine the
        .keys property. This will list the names of all attributes attached to
        the instance.
    INGESTING DICTIONARIES
        The Vessel object also allows for the ingestion of large dictionaries.
        This is useful for saving all variables in the local namespace. As an
        example:
            >>> ignore_vars = locals()
            >>> x = 42; y = np.sin(pi/3); z = np.arange(0,5,0.1)
            >>> v = Vessel('kitchen_sink.data)
            >>> v.ingest(locals(), ignore_vars)
            >>> v.save()

        We have now grabbed all variables from the local scope and saved them
        to disk. We can reconstitute this scope at a later time as follows:

            >>> w = Vessel('kitchen_sink.data') # loads data if the file exists
            >>> for key in w.keys:
            >>>     exec('%s=w.%s') % (key,key)

        The previously saved local scope will now be reconstituted.
    """

    def __init__(self, filename=None):
        self._filename = filename
        if self._filename:
            # If filename specified, and file exists, load it.
            if len(glob.glob(filename)) > 0:
                self.load()

    def _set_filename(self, filename):
        """Set the object's filename. If filename does not exist, throw an
        error."""
        if filename:
            self._filename = filename
        if not self._filename:
            raise ValueError("No filename specified.")

    def ingest(self, var_dict, ignore_variable_names=None):
        """Ingest a dictionary of variables (such as locals(), e.g.). Only
        variables in the supplied (or default) white list will be retained.
        Variables are added as attributes to the object.  """
        if ignore_variable_names:
            self.ignore_variable_names = ignore_variable_names
        else:
            self.ignore_variable_names = []
        for key in var_dict.keys():
            if key not in self.ignore_variable_names:
                self.__dict__[key] = var_dict[key]

    @property
    def keys(self):
        keys = list(self.__dict__.keys())
        keys.remove("_filename")  # don't show internal filename.
        keys.sort()
        keys.append("current_filename")
        return keys

    @property
    def current_filename(self):
        return self._filename

    def save(self, filename=None):
        """Save the data into a file with the specified name."""
        self._set_filename(filename)
        f = open(self._filename, "wb")
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, filename=None):
        """Load object from specified file."""
        self._set_filename(filename)
        f = open(self._filename, "rb")
        loaded_object = pickle.load(f)
        f.close()
        # Unpack the object and add variables as properties to this object.
        for key, val in loaded_object.items():
            self.__dict__[key] = val
