from .stage import stage
def launch_stage(stage:stage):
    total_steps=stage.steps
    if not stage.potential_params["dynamic"]["activate"]:
        for i in range(total_steps):
            stage.psi=stage.propagator(stage) # I hope this is legal
            if i%stage.monitor_params["print_scalar_interval"]:
                continue
            if i%stage.monitor_params["show_nk_interva"]:
                continue
            if i%stage.monitor_params["show_TOF_image_interval"]:
                continue
            if i%stage.monitor_params["show_verticle_TOF_interval"]:
                continue
            if i%stage.monitor_params["show_TOF_slice_interval"]:
                continue
            if i%stage.monitor_params["show_verticle_TOF_slice_interval"]:
                continue
            if i%stage.monitor_params["show_magnifier_interva"]:
                continue
            if i%stage.monitor_params["show_magnifier_slice_interval"]:
                continue
            if i%stage.monitor_params["show_phase_interval"]:
                continue

            if i%stage.save_params["scalar_interval"]:
                continue
            if i%stage.save_params["nk_interval"]:
                continue
            if i%stage.save_params["psi_interval"]:
                continue
            if i%stage.save_params["checkpoint_interval"]:
                continue

    else:
        for i in range(total_steps):
            t=i*stage.dt
            stage.psi=stage.propagator(stage,t) # I hope this is legal
            if i%stage.monitor_params["print_scalar_interval"]:
                continue
            if i%stage.monitor_params["show_nk_interva"]:
                continue
            if i%stage.monitor_params["show_TOF_image_interval"]:
                continue
            if i%stage.monitor_params["show_verticle_TOF_interval"]:
                continue
            if i%stage.monitor_params["show_TOF_slice_interval"]:
                continue
            if i%stage.monitor_params["show_verticle_TOF_slice_interval"]:
                continue
            if i%stage.monitor_params["show_magnifier_interva"]:
                continue
            if i%stage.monitor_params["show_magnifier_slice_interval"]:
                continue
            if i%stage.monitor_params["show_phase_interval"]:
                continue

            if i%stage.save_params["scalar_interval"]:
                continue
            if i%stage.save_params["nk_interval"]:
                continue
            if i%stage.save_params["psi_interval"]:
                continue
            if i%stage.save_params["checkpoint_interval"]:
                continue
