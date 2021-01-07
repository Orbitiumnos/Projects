import urllib.request
import lxml.etree as etree
from datetime import date


def get_xml(url):
    with urllib.request.urlopen(url) as url:
        s = url.read()
    return s

if __name__ == '__main__':

    xml = get_xml('http://www.cbr.ru/scripts/XML_daily.asp')
    xml_data = etree.fromstring(xml)

    today = xml_data.xpath("/ValCurs")[0].attrib['Date']
    hkd_rub = xml_data.xpath("/ValCurs/Valute[@ID='R01200']/Value")[0].text

    print('Курс гонконгского доллара к рублю на ' + today + ': ' + hkd_rub)