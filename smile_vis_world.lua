-- World file for smile_vis project

do
	matrix = require "scripts.matrix"
	actor_prototype = require "smile_vis.smile_vis_actor"

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

		ddLib.print( "smile_vis init called." )
	end

	function smile_vis:update( event, args, num_args )
		if event == "update" then
			-- Push level-specific update event
			dd_push( {event_id = level_tag} )
		
		-- initialize level graphics stuff
		if not graphics_loaded then load_graphics() end
		end
	end

end
