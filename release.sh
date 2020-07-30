#! /bin/bash

docker build -f Dockerfile -t nickalonso/ctnr-bbox-objdet:r7.0.1-ocitest .

if [ "$?" -eq "0" ]
then
	echo "Docker Build Successful, Publishing container publicly"
	docker push nickalonso/ctnr-bbox-objdet:r7.0.1-ocitest
else
 	echo "Docker Build Failed, no release executed"
fi

