repo = input("Enter an HTTPS .git repo URL to clone: ")
workers = int(input("How many workers would you like to run? (min 0, max 254): "))

template_str = open("template_start.yml", "r").read()
template_str = template_str.replace("<REPOSITORY>", repo)

default_worker_str = open("template_worker.yml", "r").read()
default_worker_str = default_worker_str.replace("<REPOSITORY>", repo)

for i in range(workers):
    worker_str = default_worker_str.replace("<WORKER_ID>", str(i))
    template_str += worker_str

print(template_str)
with open("template.yml", "w") as f:
    f.write(template_str)
