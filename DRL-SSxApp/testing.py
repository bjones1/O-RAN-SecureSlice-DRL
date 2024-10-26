from xapp_interface import ConfInterface




conf_i = ConfInterface()

imsi = conf_i.get_slice("slow")["ues"][0]

print(conf_i.unbind_ue(imsi, "slow"))
print(conf_i.unbind_ue(imsi, "fast"))
print(conf_i.unbind_ue(conf_i.get_slice("fast")["ues"][0], "fast"))
print(conf_i.bind_ue_to_slice(imsi, "fast"))

print(conf_i.get_slices())
