import fileinput

# the version of the release
with open("version.txt") as f:
    version_str = f.read()


def parse_version_str(str):
    return tuple(map(int, (str.split("."))))


version_major = str(parse_version_str(version_str)[0])
version_minor = str(parse_version_str(version_str)[1])
version_patch = str(parse_version_str(version_str)[2])
print("Parsed version:")
print("    version_major: " + version_major)
print("    version_minor: " + version_minor)
print("    version_patch: " + version_patch)

print("Updating the version in the header \"JustODE.hpp\" ...")
path_to_header = "../include/JustODE/JustODE.hpp"
is_version_changed = False
new_content = ""
for line in fileinput.input([path_to_header]):
    if line.startswith("#define JUST_ODE_VERSION_MAJOR "):
        version_major_define = "#define JUST_ODE_VERSION_MAJOR " + version_major + "\n"
        if line != version_major_define:
            is_version_changed = True
        new_content += version_major_define
    elif line.startswith("#define JUST_ODE_VERSION_MINOR "):
        version_minor_define = "#define JUST_ODE_VERSION_MINOR " + version_minor + "\n"
        if line != version_minor_define:
            is_version_changed = True
        new_content += version_minor_define
    elif line.startswith("#define JUST_ODE_VERSION_PATCH "):
        version_patch = "#define JUST_ODE_VERSION_PATCH " + version_patch + "\n"
        if line != version_patch:
            is_version_changed = True
        new_content += version_patch
    else:
        new_content += line

if is_version_changed:
    header_file = open(path_to_header, "w")
    header_file.write(new_content)
    header_file.close()
    print("The version has been modified")
else:
    print("The version has not changed")
