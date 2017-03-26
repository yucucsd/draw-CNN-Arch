import pygame


def draw_convnet(convLayer, channel, kernel, ratio, num_fully, distance, image_size, kernel_size, fully_size):
    #initialize the game engine
    pygame.init()

    #define some colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    #set width and height of the screen
    size = [1000, 500]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("ConvNet Architecture")

    #loop until user click close button
    done = False

    #used to manage how fast screen update
    clock = pygame.time.Clock()

    while done == False:
        #all event processing should go bellow this comment
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        #all event processing should go above this comment

        #all game logic should go below this comment

        #all game logic should go above this comment

        #all code to draw should go below this comment
        screen.fill(white)
        last_h = 0
        last_v = 0
        font = pygame.font.Font(None, 20)
        for i in range(len(convLayer)):#draw convolution layer
            pygame.draw.polygon(screen, black, convLayer[i], 3)
            c_pixel = [convLayer[i][2], convLayer[i][3],
            [convLayer[i][3][0] + channel[i] * ratio, convLayer[i][3][1]],
            [convLayer[i][2][0] + channel[i] * ratio, convLayer[i][2][1]]]
            pygame.draw.polygon(screen, black, c_pixel, 3)
            t_pixel = [convLayer[i][0], convLayer[i][3],
            [convLayer[i][3][0] + channel[i] * ratio, convLayer[i][3][1]],
            [convLayer[i][0][0] + channel[i] * ratio, convLayer[i][0][1]]]
            pygame.draw.polygon(screen, black, t_pixel, 3)
            text = font.render(str(channel[i]), True, black)
            screen.blit(text, [c_pixel[0][0], c_pixel[0][1] + 10])
            text = font.render(str(image_size[i]), True, black)
            screen.blit(text, [convLayer[i][1][0] - 25, convLayer[i][1][1] - 20])
            if i != len(convLayer) - 1:
                text = font.render("MAX Pooling", True, black)
                screen.blit(text, [c_pixel[3][0] + 0.5 * distance, c_pixel[3][1]])
            last_h = t_pixel[2][0]
            last_v = t_pixel[2][1]
        width = 30
        height = 200
        line_fully = []
        line_fully.append([last_h, last_v])
        for i in range(num_fully):#draw fully connected layer
            fully_start = [last_h + (i + 1) * distance + i * width, convLayer[0][0][1]]
            pygame.draw.rect(screen, black, fully_start + [width] + [height], 3)
            line_fully.append(fully_start)
            line_fully.append([fully_start[0] + width, fully_start[1]])
            text = font.render(str(fully_size[i]), True, black)
            screen.blit(text, [fully_start[0], fully_start[1] + height + 10])
            text = font.render("dense", True, black)
            screen.blit(text, [fully_start[0] - 0.9 * distance, fully_start[1] - 15])
        line = []
        for i in range(len(kernel)): # draw kernel
            pygame.draw.polygon(screen, black, kernel[i], 3)
            c_pixel = [kernel[i][2], kernel[i][3],
            [kernel[i][3][0] + channel[i] * ratio, kernel[i][3][1]],
            [kernel[i][2][0] + channel[i] * ratio, kernel[i][2][1]]]
            pygame.draw.polygon(screen, black, c_pixel, 3)
            t_pixel = [kernel[i][0], kernel[i][3],
            [kernel[i][3][0] + channel[i] * ratio, kernel[i][3][1]],
            [kernel[i][0][0] + channel[i] * ratio, kernel[i][0][1]]]
            pygame.draw.polygon(screen, black, t_pixel, 3)
            line.append([t_pixel[0], t_pixel[2], c_pixel[3]])
            text = font.render(str(kernel_size[i]), True, black)
            screen.blit(text, [kernel[i][1][0], kernel[i][1][1] + 10])
        line.append([convLayer[len(convLayer) - 1][0]])
        for i in range(len(line) - 1):
            for k in range(len(line[i])):
                pygame.draw.line(screen, black, line[i][k], line[i + 1][0], 1)
        i = 0
        while i < len(line_fully) - 2:
            pygame.draw.line(screen, black, line_fully[i], line_fully[i + 1], 1)
            i += 2
        pygame.display.flip()
        pygame.image.save(screen, "screenshot.jpg")
        #all code to draw should go above this comment
        clock.tick(20)
    pygame.quit()

if __name__ == '__main__':
    num_fully = 2
    num_conv = 4
    convLayer = []
    channel = []
    kernel = []
    channel = [3, 16, 16, 32]
    image_size = [256, 128, 64, 32]
    fully_size = [32, 3]
    kernel_size = [5, 5, 5]
    luf_h = 30 #left up point of the first layer
    luf_v = 150
    size = [200, 100, 50, 25]
    slope = [100, 50, 25, 13]
    ratio = 2
    distance = 50
    #initialize the pixel index of the convlution layer
    for i in range(num_conv):
        lu_distance = 0
        for k in range(i):
            lu_distance += channel[k] * ratio + distance + slope[k]
        lu = [luf_h + lu_distance, luf_v]
        lb = [luf_h + lu_distance, luf_v + size[i]]
        ru = [luf_h + lu_distance + slope[i], luf_v + slope[i]]
        rb = [luf_h + lu_distance + slope[i], luf_v + slope[i] + size[i]]
        convLayer.append([lu, lb, rb, ru])
    #initialize the parameters of kernel
    for i in range(num_conv - 1):
        lu_distance = 0
        for k in range(i):
            lu_distance += channel[k] * ratio + distance + slope[k]
        lu = [luf_h + lu_distance + 0.1 * slope[i], luf_v + 0.1 * slope[i]]
        lb = [luf_h + lu_distance + 0.1 * slope[i], luf_v + 0.1 * slope[i] + 0.4 * size[i]]
        ru = [luf_h + lu_distance + 0.4 * slope[i], luf_v + 0.4 * slope[i]]
        rb = [luf_h + lu_distance + 0.4 * slope[i], luf_v + 0.4 * slope[i] + 0.4 * size[i]]
        kernel.append([lu, lb, rb, ru])
    #initialize the number of kernels in convolution layer
    print convLayer
    print kernel

    if len(kernel) != len(convLayer) - 1:
        print "Num of convLayer must equal to num of kernel"
        exit()
    draw_convnet(convLayer, channel, kernel, ratio, num_fully, distance, image_size, kernel_size, fully_size)