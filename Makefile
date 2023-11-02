
# see: https://medium.com/@techhara/git-clone-private-repo-during-docker-build-8e1444bca1f3
safe-build:
	sudo docker build -t safe --build-arg ssh_prv_key="$$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$$(cat ~/.ssh/id_rsa.pub)" .

safe-run:
	sudo docker run -v "$$(pwd):/home/user/safe_and_smooth" -it safe _scripts/generate_test_results.sh

safe-bash:
	sudo docker run -v "$$(pwd):/home/user/safe_and_smooth" -it safe bash
