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

	-- Red material
	smile_vis_assets.red_mat = ddMat.new("mat_red")
	smile_vis_assets.red_mat:set_base_color(0.5, 0.0, 0.0, 1.0)
	smile_vis_assets.red_mat:set_specular(0.3)

	-- Empty camera agent
	smile_vis_assets.nav_agent = ddAgent.new("smile_vis_agent", 1.0, "kinematic")
	smile_vis_assets.nav_agent:set_pos(0.0, 0.0, 5.0)
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
