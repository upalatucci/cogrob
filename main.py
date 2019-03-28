import os
from ObjectRecognitionFramework import ObjectRecognitionFramework
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_DIR = os.path.join(ROOT_DIR, "trailer1.mp4")

def execute(orf):
    scelta=input('Inserisci\n1) Per un immagine casuale \n2) un immagine specifica \n3) Salvare un video e prendere un frame casuale del video \n4) Frame alla volta video\n5) Quit\nInserimento: ')
    if(scelta=='1'):
        #Prende casualmente una immagine dalla cartella images
        orf.input_image(IMAGE_DIR)
        return 1
    elif(scelta=='2'):
        #Inserisci il nome del file e prende quel file dalla cartella se presente
        IMAGE_SINGLE = "car.jpg"
        orf.input_single_image(IMAGE_DIR,IMAGE_SINGLE)
        return 1
    elif(scelta=='3'):
        #Acquisisce video, lo salva in una cartella e preleva una immagine casuale
        orf.input_video(VIDEO_DIR)
        return 1
    elif(scelta=='4'):
        orf.input_move(VIDEO_DIR)
        return 1
    elif scelta == '5':
        return 0
    else:
        print('Errore inserimento\n')
    print('Operazione terminata')
    return 1


if __name__ == "__main__":
    orf=ObjectRecognitionFramework()
    while(1):
        ret = execute(orf)
        if ret == 0:
            break