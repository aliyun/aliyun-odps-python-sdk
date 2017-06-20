set -e

docker pull $DOCKER_IMAGE
export PYVER=$PYVER
docker run --rm -e "PYVER=$PYVER" -v `pwd`:/io $DOCKER_IMAGE /io/bin/travis/linux-wheel.sh
