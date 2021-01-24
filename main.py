import cv2
import numpy as np
from tqdm import tqdm
import sys


class AsciiConverter:

    def __init__(self, source_video, target_video, font='fonts/AnkaCoder-C75-r.ttf'):
        self.source_video = cv2.VideoCapture(source_video)
        self.total_frames = int(self.source_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.target_video = target_video
        self.fps = self.source_video.get(cv2.CAP_PROP_FPS)

        self.ascii_chars = np.array(list(' .",:;!~+-xmo*#W&8@'))
        self.brightness_coef = 255 // (len(self.ascii_chars) - 1)

        self.font = cv2.freetype.createFreeType2()
        self.font.loadFontData(fontFileName=font, id=0)
        self.font_size = 15

        self.max_char_height = self.font.getTextSize(''.join(self.ascii_chars),
                                self.font_size, -1)[0][1]
        self.line_offset = 1
        self.resize_coef = round(125e3 / (self.source_video.get(4) * self.source_video.get(3)), 2)
        self.img_height = int(self.source_video.get(4) * self.resize_coef * (self.max_char_height + self.line_offset))
        self.img_width = None


    def get_image(self):

        ret, img = self.source_video.read()
        if not ret:
            exit()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = self.get_bg_mask(img)
        img = cv2.bitwise_and(img, img, mask=mask)
        # print(img.shape)
        img = cv2.resize(img,(0,0), fx=self.resize_coef, fy=self.resize_coef)
        # print(img.shape)

        # aspect ratio gets messed up here
        img = self.get_ascii_img(img)
        # print(img.shape) 
        return img


    def get_bg_mask(self, img):
        img = cv2.GaussianBlur(img, (11, 11), 0)
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) < 2:
            return
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.fillPoly(img, [cnts[0]], [255,255,255])
        return img

    def get_ascii_img(self, img):
        ''' Converts pixels to ASCII chars according to their brightness '''
        y = 0
        img //= self.brightness_coef
        ascii_img = None

        for row in img:
            row = ''.join(self.ascii_chars[row])

            if self.img_width is None:
                self.img_width = int(self.font.getTextSize(row,
                                    self.font_size,
                                    -1)[0][0])
            if ascii_img is None:
                ascii_img = np.ones((self.img_height,
                                     self.img_width, 3), dtype=np.uint8) * 255

            self.font.putText(img=ascii_img,
                               text=row,
                               org=(self.line_offset, y),
                               fontHeight=self.font_size,
                               color=(0,0,0),
                               thickness=-1,
                               line_type=cv2.LINE_AA,
                               bottomLeftOrigin=False)

            y += self.line_offset + self.max_char_height
            
        return ascii_img

    def run(self):
        pbar = tqdm(total=self.total_frames)
        a = 0
        while self.source_video.isOpened():
            try:
                img = self.get_image()
                img = cv2.resize(img,(self.img_height,
                                      self.img_width))

                if isinstance(self.target_video, str):
                    self.target_video = cv2.VideoWriter(self.target_video,
                                                cv2.VideoWriter_fourcc(*'avc1'),
                                                self.fps,
                                                (self.img_height,
                                                 self.img_width))
                self.target_video.write(img)
                pbar.update()

            except KeyboardInterrupt:
                break        

        pbar.close()
        self.target_video.release()


if __name__ == '__main__':
    source, output = sys.argv[1], sys.argv[2]
    app = AsciiConverter(source, output)
    app.run()
