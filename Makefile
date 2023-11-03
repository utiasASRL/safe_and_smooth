
# see: https://medium.com/@techhara/git-clone-private-repo-during-docker-build-8e1444bca1f3
safe-build:
	sudo docker build -t safe .

# run test and generate test results
safe-test:
	sudo docker run -v "$$(pwd):/home/user/safe_and_smooth" -it safe sh -c "pytest ."
	sudo docker run -v "$$(pwd):/home/user/safe_and_smooth" -it safe _scripts/generate_test_results.sh

# generate all results (in background)
safe-run:
	sudo docker run -v "$$(pwd):/home/user/safe_and_smooth" -dt safe _scripts/generate_all_results.sh

# run docker container in interactive bash mode
safe-bash:
	sudo docker run -v "$$(pwd):/home/user/safe_and_smooth" -it safe bash
