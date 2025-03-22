import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
pygame.init()

# Initialize pygame

WINDOWSIZEX = 800
WINDOWSIZEY = 600

BOUNDARYINC =  5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

FONT = pygame.font.Font("freesansbold.ttf", 18)

IMAGESAVE = False

MODEL = load_model("bestmodel.keras")

LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5: "Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption('Draw a Digit')

iswriting = False

number_xcord = []
number_ycord = []

image_cnt = 1
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            # Sort the coordinates
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            # Calculate the bounding box correctly
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

            # Clear the coordinate lists after use
            number_xcord = []
            number_ycord = []

            # Draw the red rectangle around the drawn digit
            pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

            # Capture the area of the digit
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)  # Padding for better prediction
                image = cv2.resize(image, (28, 28)) / 255

                # Predict the label from the model
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                # Render the label text
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                # Display the predicted label
                DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)  # Clear screen

    pygame.display.update()
