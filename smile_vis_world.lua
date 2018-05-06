-- World file for smile_vis project

do
	matrix = require "scripts.matrix"
	actor_prototype = require "smile_vis.smile_vis_actor"
	data_m = require "smile_vis.smile_vis_data_manager"

	smile_vis = {}

	graphics_loaded = false


	level_tag = "smile_vis_update"

	function smile_vis:init( event, args, num_args )
		-- Make assets object local
		assets = smile_vis_assets

		-- Create actor controller and set variables
		smile_vis.actor = actor_prototype:new( {name = "actor_01"} )
		-- Register actor callback
		dd_register_callback(smile_vis.actor.name, smile_vis.actor)
		-- Subscribe to smile_vis_world generated callback
		dd_subscribe( {key = smile_vis.actor.name, event = level_tag} )
		
		-- subscribe data manager
		data_m.name = "smile_data_manager"
		data_m.data = SController:get()		
		data_m.data.tile = 0.5
		dd_register_callback(data_m.name, data_m)
		dd_subscribe( {key = data_m.name, event = level_tag} )
		dd_subscribe( {key = data_m.name, event = "calc_offset"} )

		-- log screen dimensions
		assets.scr_x, assets.scr_y = ddLib.scr_dimensions()

		-- open folder containing smile data
		load_folder(PROJECT_DIR.."/smile_vis/input")
		groundtruth_folder(PROJECT_DIR.."/smile_vis/ground_truth")
		w_b_folders(PROJECT_DIR.."/smile_vis/weight",PROJECT_DIR.."/smile_vis/bias")

		ddLib.print( "smile_vis init called." )
	end

	function smile_vis:update( event, args, num_args )
		if event == "update" then
			-- Push level-specific update event
			dd_push( {event_id = level_tag} )
			-- show ui
			smile_UI()
		
		-- initialize level graphics stuff
		if not graphics_loaded then load_graphics(); graphics_loaded = true; end
		end
	end

end
