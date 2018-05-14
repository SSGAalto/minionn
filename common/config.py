"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""
# Fractions for shifting up the input data and scaling down after every layer (to prevent overflows)
fractional_base = 1000
fractional_downscale = 1

# MiniONN static vars
SLOTS = 4096
PMAX = 101285036033
PMAX_HALF = int(PMAX / 2)

# Key files
asset_folder = "assets/"
client_pkey = asset_folder + "c.pkey"
client_skey = asset_folder + "c.skey"
server_pkey = asset_folder + "s.pkey"
server_skey = asset_folder + "s.skey"

# Server config, default values
ip = "127.0.0.1"
port_rpc = 8555
port_aby = 8556

# GRPC options
grpc_options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]

# Debugging parameters
debug_mode = False
debug_print_length = 5
debug_print_length_long = 10

random_r = True
random_v = True