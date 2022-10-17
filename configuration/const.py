import logging.config


logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Pytorch Image Template')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
