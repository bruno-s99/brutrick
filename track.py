
'''
get position (boundingbox) in image ---> position of next search based on velocity and direction of drone
top left and bottom right corner should be passed
'''


#right now: w,h  x,y
image_size = (1920,1080)

####boundingbox one is detected in frame i-1, boundingboxtwo ind frame i-2. For testing, this as example:
boundingbox_one = (1850,160,1900,350)
boundingbox_two = (1820,210,1870,400)


def get_size(bb):
    tl_x,tl_y,br_x,br_y = bb
    return (br_x - tl_x), (br_y - tl_y)

def get_direction(bb_one, bb_two):
    #x,y,z - Richtung, skaling of x,y,z is velocity
    z = ((get_size(bb_two)[0] - get_size(bb_one)[0]) + (get_size(bb_two)[1] - get_size(bb_one)[1])) / 4
    y = ((bb_two[1] - bb_one[1]) + (bb_two[3] - bb_one[3])) / 2
    x = ((bb_two[0] - bb_one[0]) + (bb_two[2] - bb_one[2])) / 2
    
    return x,y,z



def new_coords(bb,image_size,ratio = 1.5, bb_two = 0):
    
    tl_x,tl_y,br_x,br_y = bb    

    if bb_two != 0:
        #different ratio
        x,y,z = get_direction(bb_two, bb)
        size_x, size_y = get_size(bb)  # predict 
        size_x = size_x * (1 + (z/100))
        size_y = size_y * (1 + (z/100))
        
        new_img_tl_corner_x = tl_x + x - ratio*size_x
        new_img_tl_corner_y = tl_y + y - ratio*size_y

        new_img_br_corner_x = br_x + x + ratio*size_x
        new_img_br_corner_y = br_y + y + ratio*size_y
       
        if tl_x + x < ratio*size_x:
            new_img_tl_corner_x = 0
        
        if tl_y + y < ratio*size_y:
            new_img_tl_corner_y = 0

        if new_img_br_corner_x > image_size[0]:
            new_img_br_corner_x=image_size[0]-1
        
        if new_img_br_corner_y > image_size[1]:
            new_img_br_corner_y=image_size[1] - 1

    else:
        
        size_x, size_y = get_size(bb)
        new_img_tl_corner_x = tl_x - ratio*size_x
        new_img_tl_corner_y = tl_y - ratio*size_y

        new_img_br_corner_x = br_x + ratio*size_x
        new_img_br_corner_y = br_y + ratio*size_y
    #corner cases
        if tl_x < ratio*size_x:
            new_img_tl_corner_x = 0
            
        if tl_y < ratio*size_y:
            new_img_tl_corner_y = 0
            
        if image_size[0] - br_x < ratio*size_x:
            new_img_br_corner_x=image_size[0]-1

        if image_size[1] - br_y < ratio*size_y:
            new_img_br_corner_y=image_size[1] - 1
            
    

    print('tlx:',new_img_tl_corner_x)
    print('tly:',new_img_tl_corner_y)
    print('brx:',new_img_br_corner_x)
    print('bry:',new_img_br_corner_y)

    return new_img_tl_corner_x, new_img_tl_corner_y, new_img_br_corner_x, new_img_br_corner_y

#new_coords(boundingbox_one, image_size, bb_two = boundingbox_two)
#new_coords(boundingbox_one, image_size)
