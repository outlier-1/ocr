# import the necessary packages
import cv2
import math
print(math.fabs(-3))
def detect_text(realim, blueprint):
    img = blueprint

    biaspos=5
    biasneg=5
    coor1=0
    coor2=0
    coor3=0
    coor4=0

    x=-1
    s=-1
    coordlist=[]
    while x<1050:
        x+=1
        print("start")
        if 255 in img[x,:,:]:
            print("ilk beyaz satır bulundu.." + str(x))
            coor1=x
            while True:
                x+=1
                if 255 not in img[x,:,:]:
                    print("ikinci beyaz satır bulundu.." + str(x))
                    coor2=x
                    break

            while s<749:
                s+=1
                if 255 in img[coor1:coor2, s, :]:
                    coor3=s
                    while True:
                        s+=1
                        if 255 not in img[coor1:coor2, s, :]:
                            coor4=s
                            if int(math.fabs(coor1-coor2)) / int(math.fabs(coor3-coor4)) > 5:
                                break
                            else:
                                coordlist.append([coor1,coor2,coor3,coor4])
                            break
            s=-1

    for i in range(0, len(coordlist)):
        cv2.rectangle(realim, ((coordlist[i][2])+biasneg, coordlist[i][0]),
                  ((coordlist[i][3])+biaspos, (coordlist[i][1])),
                  (0,255,0), 2)
    return realim
