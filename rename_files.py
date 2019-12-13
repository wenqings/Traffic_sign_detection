import os

path = 'G:\\RSA_GIT\\RSA_TomTom\\Traffic_sign_detection_Machine_learning\\signs_stage2'
count = 0
last_folder = None
for root, dirs, files in os.walk(path):
        for i in files:

            file_path = os.path.join(root, i)
            if '.png' in file_path or '.PNG' in file_path:
                folder = file_path.split('\\')[-2]
                if folder!= last_folder:
                    count = 0
                    last_folder = folder
                print(file_path)
                print(folder)
                print('new_path:',root+'\\'+folder+'_'+str(count)+'.png')
                count+=1
                try:
                    # If file already exist, skip
                    os.rename(file_path,root+'\\'+folder+'_'+str(count)+'.png')
                except:
                    pass
