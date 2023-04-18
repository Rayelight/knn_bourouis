import sys

allLabels = ['0','1','2','3']
nbLines = 4000
fd = open(sys.argv[1],'r')
lines = fd.readlines()

count=0
for label in lines:
	if label.strip() in allLabels:
		count+=1
	else:
		if count<nbLines:
			print("Wrong label line:"+str(count+1))
			break
if count<nbLines:
	print("Labels Check : fail!")
else:
	print("Labels Check : Successfull!")


