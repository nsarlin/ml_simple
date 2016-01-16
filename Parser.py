import struct
import numpy as np

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