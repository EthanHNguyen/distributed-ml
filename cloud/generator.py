repo = input("Enter an HTTPS .git repo URL to clone: ")
workers = int(input("How many workers would you like to run? (min 0, max 254): "))
working_dir = input("What directory is your requirements.txt in? (. for root directory of repo): ")
entrypoint = input("What Python file is your entrypoint in, relative to that directory? (e.g. main.py): ")

template_str = open("template_start.yml", "r").read()
template_str = template_str.replace("<REPOSITORY>", repo)
template_str = template_str.replace("<NUM_NODES>", str(workers + 1))
template_str = template_str.replace("<WORKING_DIR>", working_dir)
template_str = template_str.replace("<ENTRYPOINT>", entrypoint)

default_worker_str = open("template_worker.yml", "r").read()
default_worker_str = default_worker_str.replace("<REPOSITORY>", repo)
default_worker_str = default_worker_str.replace("<NUM_NODES>", str(workers + 1))
default_worker_str = default_worker_str.replace("<WORKING_DIR>", working_dir)
default_worker_str = default_worker_str.replace("<ENTRYPOINT>", entrypoint)


for i in range(workers):
    worker_str = default_worker_str.replace("<WORKER_ID>", str(i + 1))
    template_str += worker_str

with open("template.yml", "w") as f:
    f.write(template_str)

print("Generated at template.yml!")