from PIL import Image

import pyocr
import sys
import pyocr.builders

tools = pyocr.get_available_tools()
print (len(tools))
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print ("available tools: '%s' , '%s' , '%s"'')
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))

img  = Image.open('scan001.jpg')


if tool.can_detect_orientation():
    orientation = tool.detect_orientation(
        img,
        lang='rus'
    )
    print("Orientation: {}".format(orientation))

print(tool.image_to_string(
    img,
    lang="rus",
    builder=pyocr.builders.TextBuilder()
))