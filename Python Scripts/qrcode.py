

'''
def main():

    qr = qrcode.QRCode(
	    version=1,
	    box_size=15,
	    border=5
    )

    data = input()
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save('qr_python.png')
    img.show()
    return
if __name__ == '__main__':
    main()


qr = qrcode.QRCode(version=1,box_size=15,border=5)
data = 'Nick Fedorov'
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill='black', back_color='white')
img.save('qr_python.png')
'''

import qrcode
qr = qrcode.make('hello world')
#img.show()
qr.save('qr_python.png')