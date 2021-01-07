#import webbrowser
#webbrowser.open_new(r'C:\Users\Nikolay\Documents\SCRIPTS\it_spend.pdf')

import os
import tempfile
from pdf2image import convert_from_path

filename = r'C:/Users/Nikolay/Documents/SCRIPTS/it_spend.pdf'

with tempfile.TemporaryDirectory() as path:
     images_from_path = convert_from_path(filename, output_folder=path, last_page=1, first_page =0)

base_filename  =  os.path.splitext(os.path.basename(filename))[0] + '.jpg'

save_dir = 'C:/Users/Nikolay/Documents/SCRIPTS'

for page in images_from_path:
    page.save(os.path.join(save_dir, base_filename), 'JPEG')


from pdf2image import convert_from_bytes
images = convert_from_bytes(open('C:/Users/Nikolay/Documents/SCRIPTS/it_spend.pdf', 'rb').read())
images_from_path = convert_from_path(filename, output_folder=path, last_page=1, first_page =0)