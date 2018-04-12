-- Asset file for smile_vis project

smile_vis_assets = {}

function load()
	arg = {}

	-- Cube mesh
	smile_vis_assets.cube_m = ddModel.new(ROOT_DIR.."/Resource/Meshes/primitives/cube.ddm")
	ddLib.print( "Created plane mesh: ", smile_vis_assets.cube_m:id() )

	-- Directional light
	smile_vis_assets.light_1 = ddLight.new("light_1")
	smile_vis_assets.light_1:set_active(true)
	ddLib.print( "Created light: ", smile_vis_assets.light_1:id() )

	-- Floor agent
	smile_vis_assets.floor = ddAgent.new("floor_agent", 0.0, "box")
	smile_vis_assets.floor:set_scale(100.0, 0.2, 100.0)
	smile_vis_assets.floor:add_mesh(smile_vis_assets.cube_m:id(), 0.1, 100.0)
	ddLib.print( "Created agent (floor): ", smile_vis_assets.floor:id() )

	-- Red material
	smile_vis_assets.red_mat = ddMat.new("mat_red")
	smile_vis_assets.red_mat:set_base_color(0.5, 0.0, 0.0, 1.0)
	smile_vis_assets.red_mat:set_specular(0.3)
	-- Box agent
	smile_vis_assets.rand_obj = ddAgent.new("rand_obj", 1.0, "box")
	smile_vis_assets.rand_obj:set_pos(0.0, 5.0, 0.0)
	smile_vis_assets.rand_obj:add_mesh(smile_vis_assets.cube_m:id(), 0.1, 100.0)
	smile_vis_assets.rand_obj:set_mat_at_idx(0, 0, smile_vis_assets.red_mat:id())
	ddLib.print( "Created agent (rand_obj): ", smile_vis_assets.rand_obj:id() )

	-- Empty camera agent
	smile_vis_assets.nav_agent = ddAgent.new("smile_vis_agent", 1.0, "kinematic")
	smile_vis_assets.nav_agent:set_pos(0.0, 2.0, 5.0)
	smile_vis_assets.nav_agent:set_eulerPYR(0.0, 0.0)
	ddLib.print( "Created agent (smile_vis_agent): ", smile_vis_assets.nav_agent:id() )

	-- Attach camera
	args = {}
	args["id"] = "cam_01"
	args["parent"] = cam_agent_id
	smile_vis_assets.cam_01 = ddCam.new("cam_01", smile_vis_assets.nav_agent:id())
	smile_vis_assets.cam_01:set_active(true)
	ddLib.print( "Created camera: ", smile_vis_assets.cam_01:id() )

end
