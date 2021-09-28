import os
import describe


def main():
    dir = '/shelf/20210607_vertical_2000'
    detector    = describe.detectors('AKAZE')
    descriptors = describe.compute(detector, dir)
    describe.save(descriptors, os.path.basename(dir) + '_descriptors')


if __name__ == '__main__':
    main()