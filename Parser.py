import struct
import numpy as np
import scipy.io as sio

class IdxParser(object):
    """
    This class implements a generic idx Parser, used for importing
    matrices stored as idx files into numpy arrays
    """
    def __init__(self, filename):
        """Sets a parser up"""
        self.file = open(filename, "rb")
        self.magic_high = None
        self.shape = []


    def read(self, fmt):
        """
        Reads the file, and returns a number according to fmt.
        (cf the documentation of struct module)
        All idx files are supposed to be big endian.
        """
        size = struct.calcsize(fmt)
        tmp = self.file.read(size)
        return struct.unpack(fmt, tmp)[0]


    def read_int(self):
        """ Reads an int in the file """
        return self.read(">I")


    def read_short(self):
        """ Reads a short in the file """
        return self.read(">H")


    def read_byte(self):
        """ Reads a byte in the file """
        return self.read(">B")


    def get_magic(self):
        """
        Reads the magic number in the file.
        This magic contains the type of data stored in the matrix,
        and the dimensions of the matrix.
        """
        if self.magic_high is None:
            self.magic_high = self.read_short()
            self.data_type = self.read_byte()
            self.data_dim = self.read_byte()
        return self.magic_high, self.data_type, self.data_dim


    def get_size(self):
        """
        Reads the size of the next dimension in the file
        """
        size = self.read_int()
        self.shape.append(size)


    def parse_data(self):
        """
        Reads the matrix and stores it in a numpy array
        """
        for i in range(self.data_dim):
            self.get_size()
        databytes = self.file.read(np.prod(self.shape))
        matrix = np.frombuffer(databytes, self.data_fmt)
        matrix = matrix.reshape(self.shape)
        return matrix


    def parse(self):
        """
        Reads the whole file and returns an array.
        """
        # Parse generic header
        self.get_magic()

        if self.magic_high != 0:
            raise ValueError("Wrong filetype")
        # Get data format
        if self.data_type == 0x8:
            self.data_fmt = ">B"
        elif self.magic == 0x9:
            self.data_fmt = ">b"
        elif self.magic == 0xb:
            self.data_fmt = ">h"
        elif self.magic == 0xc:
            self.data_fmt = ">i"
        elif self.magic == 0xd:
            self.data_fmt = ">f"
        elif self.magic == 0xe:
            self.data_fmt = ">d"
        else:
            raise ValueError("Wrong filetype")

        return self.parse_data()


class CsvParser(object):
    def __init__(self, filename, delim=','):
        """Sets a parser up"""
        self.file = open(filename, "r")
        self.delim = delim
        self.find_xshape()


    def find_xshape(self):
        """
        xshape is the number of elements in the first line of the csv file
        We will use it to check that the size is consistent throughout the file.
        yshape is simply the number of lines.
        """
        line = self.file.readline()
        self.xshape = line.count(self.delim) + 1
        self.file.seek(0)


    def parse_lines(self):
        """
        Generator. Parse one line on each iteration, and check that its number
        of elements is correct.
        """
        for line in self.file:
            new_row = [float(x) for x in line.split(self.delim)]
            if len(new_row) != self.xshape:
                raise ValueError("Badly formated file")
            yield new_row


    def parse(self):
        """
        parses the whole file an returns an array.
        """
        matrix = np.array([])
        for row in self.parse_lines():
            if matrix.shape == (0,):
                matrix = np.insert(matrix, 0, row).reshape(self.xshape, 1)
            else:
                matrix = np.concatenate((matrix,
                                         np.array(row).reshape(self.xshape, 1)),
                                        1)
        return matrix.transpose()



class MatParser(object):
    """
    Parser for matlab .mat files, using scipy.io.
    This class is just a wrapper for scipy.io.loadmat.
    """

    def __init__(self, filename):
        self.filename = filename
        self.loaded = sio.loadmat(self.filename)

        # Prints available matrices
        print([key for key in self.loaded.keys() if
               key.startswith('__') is False])


    def parse(self, key):
        """
        .mat object can hold several matrices.
        key is the name of the matrix we want to get.
        """
        return self.loaded[key]
