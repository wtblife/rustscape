use amethyst::{
    animation::{
        get_animation_set, AnimationBundle, AnimationCommand, AnimationControlSet, AnimationSet,
        EndControl, VertexSkinningBundle,
    },
    assets::{
        AssetPrefab, AssetStorage, Completion, Handle, Prefab, PrefabData, PrefabLoader,
        PrefabLoaderSystemDesc, ProgressCounter, RonFormat,
    },
    controls::{
        ArcBallControlBundle, ArcBallControlTag, ControlTagPrefab, FlyControlBundle, HideCursor,
    },
    core::{
        geometry::Plane,
        math::{
            convert, distance_squared, Isometry3, Matrix4, Point2, Point3, Translation3,
            UnitQuaternion, Vector2, Vector3, Vector4,
        },
        shrev::{EventChannel, ReaderId},
        timing::Time,
        transform::{Transform, TransformBundle},
    },
    derive::{PrefabData, SystemDesc},
    ecs::{
        Component, DenseVecStorage, Entities, Entity, Join, NullStorage, Read, ReadExpect,
        ReadStorage, System, SystemData, World, WorldExt, Write, WriteStorage,
    },
    gltf::{GltfSceneAsset, GltfSceneFormat, GltfSceneLoaderSystemDesc},
    input::{
        is_close_requested, is_key_down, is_mouse_button_down, InputBundle, InputEvent,
        InputHandler, ScrollDirection, StringBindings,
    },
    prelude::*,
    renderer::{
        camera::{ActiveCamera, Camera, CameraPrefab},
        debug_drawing::{DebugLines, DebugLinesComponent, DebugLinesParams},
        formats::GraphicsPrefab,
        light::LightPrefab,
        palette::Srgba,
        plugins::{RenderDebugLines, RenderPbr3D, RenderShaded3D, RenderSkybox, RenderToWindow},
        rendy::mesh::{Normal, Position, Tangent, TexCoord},
        types::DefaultBackend,
        types::{Mesh, MeshData},
        visibility::{BoundingSphere, Frustum, Visibility},
        RenderingBundle,
    },
    ui::{RenderUi, UiBundle, UiCreator, UiFinder, UiText},
    utils::{
        application_root_dir,
        auto_fov::{AutoFov, AutoFovSystem},
        fps_counter::*,
        tag::{Tag, TagFinder},
    },
    window::ScreenDimensions,
    winit::{MouseButton, VirtualKeyCode},
    Error,
};

use log::{error, info};
use ncollide3d::{
    query::{PointQuery, Ray, RayCast},
    shape::{Ball, ConvexHull, TriMesh},
};
use oxygengine_navigation::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// DESIGN
//player clicks and InteractionSystem publishes MoveEntity event (which can be to an entity or position) **FUTURE: publishes event message for NetworkMessageSystem instead**
//**FUTURE: server sends MoveEntity<EntityType> event with move path (vec![Point2]) and movement speed (f32)**
//**FUTURE: NetworkMessageSystem deserializes/instantiates/publishes actual event?**
//MovementSystem processes MoveEntity events, updates the entity movement component {move_positions, movement_speed}, and moves entities toward next move_position
//AnimationSystem cycles through entities with animation tags and uses entity state (checks tags?) to update animations as necessary
//**FUTURE: MovementSystem will also process SyncEntityPosition events which the server sends periodically (before every MoveEntity event and at an interval) and move entity toward servers' position before moving player toward next move_position
//Should also sync stats like HP periodically to prevent anything weird

const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const ATTACK_INTERVAL: f32 = 1.5;
const CAMERA_SENSITIVITY_X: f32 = 0.1;
const CAMERA_SENSITIVITY_Y: f32 = 0.1;
const CAMERA_MIN_DISTANCE: f32 = 9.0;
const CAMERA_MAX_DISTANCE: f32 = 30.0;
const CURRENT_PLAYER: u32 = 2;
const CHUNK_SIZE: u32 = 100;

use std::path::PathBuf;
fn main() -> Result<(), Error> {
    amethyst::start_logger(Default::default());

    let app_dir = PathBuf::from(r"D:/playing with rust/rustscape"); //application_root_dir().unwrap();
    let display_config_path = app_dir.join("config/display.ron");
    let assets_dir = app_dir.join("assets/");

    let game_data = GameDataBuilder::new()
        .with_system_desc(
            PrefabLoaderSystemDesc::<ScenePrefabData>::default(),
            "scene_loader",
            &[],
        )
        .with_system_desc(
            GltfSceneLoaderSystemDesc::default(),
            "gltf_loader",
            &["scene_loader"], // This is important so that entity instantiation is performed in a single frame.
        )
        .with_bundle(
            AnimationBundle::<usize, Transform>::new("animation_control", "sampler_interpolation")
                .with_dep(&["gltf_loader"]),
        )?
        .with_bundle(
            ArcBallControlBundle::<StringBindings>::new()
                .with_sensitivity(CAMERA_SENSITIVITY_X, CAMERA_SENSITIVITY_Y),
        )?
        .with(AutoFovSystem::new(), "auto_fov", &["scene_loader"])
        .with_bundle(TransformBundle::new().with_dep(&[
            "animation_control",
            "sampler_interpolation",
            "free_rotation",
            "arc_ball_rotation",
        ]))?
        .with_bundle(VertexSkinningBundle::new().with_dep(&[
            "transform_system",
            "animation_control",
            "sampler_interpolation",
        ]))?
        .with_bundle(InputBundle::<StringBindings>::new())?
        .with_system_desc(
            CameraDistanceSystemDesc::default(),
            "camera_distance_system",
            &["input_system"],
        )
        .with_system_desc(
            InteractionSystemDesc::default(),
            "interaction_system",
            &[
                "transform_system",
                "input_system",
                "camera_distance_system",
                "animation_control",
                "sampler_interpolation",
            ],
        )
        .with_system_desc(
            MovementSystemDesc::default(),
            "movement_system",
            &[
                "transform_system",
                "camera_distance_system",
                "animation_control",
                "sampler_interpolation",
            ],
        )
        .with_system_desc(
            AnimationSystemDesc::default(),
            "animation_system",
            &["animation_control", "sampler_interpolation"],
        )
        .with_system_desc(
            AttackSystemDesc::default(),
            "attack_system",
            &[
                "transform_system",
                "animation_control",
                "sampler_interpolation",
            ],
        )
        .with_bundle(UiBundle::<StringBindings>::new())?
        .with_bundle(
            RenderingBundle::<DefaultBackend>::new()
                .with_plugin(
                    RenderToWindow::from_config_path(display_config_path)?.with_clear(CLEAR_COLOR),
                )
                // .with_plugin(RenderPbr3D::default().with_skinning())
                .with_plugin(RenderShaded3D::default().with_skinning())
                .with_plugin(RenderUi::default())
                .with_plugin(RenderSkybox::default())
                .with_plugin(RenderDebugLines::default()),
        )?;

    let mut game = Application::build(assets_dir, Loading::new())?.build(game_data)?;
    game.run();

    Ok(())
}

#[derive(Default, Deserialize, PrefabData, Serialize)]
#[serde(default)]
struct ScenePrefabData {
    graphics: Option<GraphicsPrefab<(Vec<Position>, Vec<Normal>, Vec<Tangent>, Vec<TexCoord>)>>,
    transform: Option<Transform>,
    light: Option<LightPrefab>,
    camera: Option<CameraPrefab>,
    auto_fov: Option<AutoFov>, // `AutoFov` implements `PrefabData` trait
    gltf: Option<AssetPrefab<GltfSceneAsset, GltfSceneFormat>>,
    animation: Option<AnimationComponent>,
    control_tag: Option<ControlTagPrefab>,
    movement: Option<MovementComponent>,
    attackable: Option<Tag<AttackableTag>>,
    interactable: Option<Tag<InteractableTag>>,
    obstacle: Option<Tag<ObstacleTag>>,
    terrain_tag: Option<Tag<TerrainTag>>,
}

fn load_nav_mesh(world: &mut World, gltf_path: &str) {
    let mut debug = DebugLinesComponent::default();

    let (gltf, buffers, _) = gltf::import(gltf_path).unwrap();
    for mesh in gltf.meshes() {
        let mut vertices: Vec<NavVec3> = vec![];
        let mut triangles: Vec<NavTriangle> = vec![];
        println!("Mesh #{}", mesh.index());
        for primitive in mesh.primitives() {
            println!("- Primitive #{}", primitive.index());

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            use gltf::mesh::util::ReadIndices;
            let mut indices: Vec<u32> = match reader.read_indices() {
                Some(ReadIndices::U8(iter)) => iter.map(u32::from).collect(),

                Some(ReadIndices::U16(iter)) => iter.map(u32::from).collect(),

                Some(ReadIndices::U32(iter)) => iter.collect(),

                None => vec![],
            };

            if let Some(iter) = reader.read_positions() {
                for vertex in iter {
                    let nav_vec: NavVec3 = vertex.into();
                    vertices.push(nav_vec);
                }
            }

            //group into triangles
            for i in 0..indices.len() {
                if (i + 1) % 3 == 0 {
                    triangles.push((indices[i], indices[i - 1], indices[i - 2]).into());
                }
            }
        }

        // for triangle in &triangles {
        //     let f = vertices[triangle.first as usize];
        //     let s = vertices[triangle.second as usize];
        //     let t = vertices[triangle.third as usize];
        //     debug.add_line(
        //         [f.x as f32, f.y as f32, f.z as f32].into(),
        //         [s.x as f32, s.y as f32, s.z as f32].into(),
        //         Srgba::new(0.0, 0.0, 0.0, 1.0),
        //     );
        //     debug.add_line(
        //         [s.x as f32, s.y as f32, s.z as f32].into(),
        //         [t.x as f32, t.y as f32, t.z as f32].into(),
        //         Srgba::new(0.0, 0.0, 0.0, 1.0),
        //     );
        //     debug.add_line(
        //         [t.x as f32, t.y as f32, t.z as f32].into(),
        //         [f.x as f32, f.y as f32, f.z as f32].into(),
        //         Srgba::new(0.0, 0.0, 0.0, 1.0),
        //     );
        // }

        let mesh = NavMesh::new(vertices, triangles).unwrap();
        world.write_resource::<NavMeshesRes>().register(mesh);
    }

    // println!("{:?}", triangles);
    // println!("{:?}", vertices);

    // world
    //     .create_entity()
    //     .with(Transform::default())
    //     .with(debug)
    //     .build();
}

struct Loading {
    progress: ProgressCounter,
    scene: Option<Handle<Prefab<ScenePrefabData>>>,
}

impl Loading {
    fn new() -> Self {
        Loading {
            progress: ProgressCounter::new(),
            scene: None,
        }
    }
}

impl SimpleState for Loading {
    fn on_start(&mut self, data: StateData<GameData>) {
        data.world.exec(|mut creator: UiCreator<'_>| {
            creator.create("ui/loading.ron", &mut self.progress);
        });

        let handle = data
            .world
            .exec(|loader: PrefabLoader<'_, ScenePrefabData>| {
                loader.load("prefab/scene_prefab.ron", RonFormat, &mut self.progress)
            });

        data.world.insert(NavMeshesRes::default());
        load_nav_mesh(data.world, "assets/mesh/terrain/TerrainTest1.glb");
        self.scene = Some(handle);
    }

    fn update(&mut self, data: &mut StateData<'_, GameData<'_, '_>>) -> SimpleTrans {
        match self.progress.complete() {
            Completion::Loading => Trans::None,
            Completion::Failed => {
                error!("Failed to load the scene");
                Trans::Quit
            }
            Completion::Complete => {
                info!("Loading finished. Moving to the main state.");
                Trans::Switch(Box::new(MyScene {
                    scene: self.scene.take().unwrap(),
                }))
            }
        }
    }
}

struct MyScene {
    scene: Handle<Prefab<ScenePrefabData>>,
}

impl SimpleState for MyScene {
    fn on_start(&mut self, data: StateData<'_, GameData<'_, '_>>) {
        data.world.create_entity().with(self.scene.clone()).build();
        data.world
            .exec(|finder: UiFinder| finder.find("loading"))
            .map_or_else(
                || error!("Unable to find Ui Text `loading`"),
                |e| {
                    data.world
                        .delete_entity(e)
                        .unwrap_or_else(|err| error!("{}", err))
                },
            );

        let mut hide_cursor = HideCursor::default();
        hide_cursor.hide = false;
        data.world.insert(hide_cursor);
    }

    fn update(&mut self, data: &mut StateData<'_, GameData<'_, '_>>) -> SimpleTrans {
        Trans::None
    }

    fn handle_event(
        &mut self,
        data: StateData<'_, GameData<'_, '_>>,
        event: StateEvent,
    ) -> SimpleTrans {
        if let StateEvent::Window(ref event) = event {
            if is_close_requested(event) || is_key_down(event, VirtualKeyCode::Escape) {
                Trans::Quit
            } else {
                Trans::None
            }
        } else {
            Trans::None
        }
    }
}

#[derive(SystemDesc)]
#[system_desc(name(CameraDistanceSystemDesc))]
struct CameraDistanceSystem {
    #[system_desc(event_channel_reader)]
    event_reader: ReaderId<InputEvent<StringBindings>>,
}

impl CameraDistanceSystem {
    pub fn new(event_reader: ReaderId<InputEvent<StringBindings>>) -> Self {
        CameraDistanceSystem { event_reader }
    }
}

impl<'a> System<'a> for CameraDistanceSystem {
    type SystemData = (
        Read<'a, EventChannel<InputEvent<StringBindings>>>,
        ReadStorage<'a, Transform>,
        WriteStorage<'a, ArcBallControlTag>,
        Write<'a, HideCursor>,
    );

    fn run(&mut self, (events, transforms, mut tags, mut hide_cursor): Self::SystemData) {
        for event in events.read(&mut self.event_reader) {
            match *event {
                InputEvent::MouseButtonPressed(MouseButton::Middle) => {
                    hide_cursor.hide = true;
                }
                InputEvent::MouseButtonReleased(MouseButton::Middle) => {
                    hide_cursor.hide = false;
                }
                InputEvent::MouseWheelMoved(direction) => {
                    for (_, tag) in (&transforms, &mut tags).join() {
                        match direction {
                            ScrollDirection::ScrollUp => {
                                tag.distance = CAMERA_MIN_DISTANCE.max(tag.distance * 0.9);
                            }
                            ScrollDirection::ScrollDown => {
                                tag.distance = CAMERA_MAX_DISTANCE.min(tag.distance * 1.1);
                            }
                            _ => (),
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

#[derive(SystemDesc)]
#[system_desc(name(InteractionSystemDesc))]
struct InteractionSystem {
    #[system_desc(event_channel_reader)]
    event_reader: ReaderId<InputEvent<StringBindings>>,
}

impl InteractionSystem {
    pub fn new(event_reader: ReaderId<InputEvent<StringBindings>>) -> Self {
        InteractionSystem { event_reader }
    }
}

impl<'a> System<'a> for InteractionSystem {
    type SystemData = (
        Read<'a, EventChannel<InputEvent<StringBindings>>>,
        Read<'a, InputHandler<StringBindings>>,
        ReadStorage<'a, Transform>,
        ReadStorage<'a, Camera>,
        Read<'a, ActiveCamera>,
        ReadExpect<'a, ScreenDimensions>,
        Read<'a, HideCursor>,
        Entities<'a>,
        Write<'a, EventChannel<MovementEvent>>,
        Write<'a, EventChannel<AttackEvent>>,
        ReadStorage<'a, BoundingSphere>,
        ReadStorage<'a, Tag<AttackableTag>>,
        ReadStorage<'a, Tag<InteractableTag>>,
        Read<'a, NavMeshesRes>,
        Read<'a, Visibility>,
        ReadStorage<'a, Tag<TerrainTag>>,
    );

    fn run(
        &mut self,
        (
            input_events,
            input_handler,
            transforms,
            cameras,
            active_camera,
            screen_dimensions,
            hide_cursor,
            entities,
            mut movement_events,
            mut attack_events,
            bounding_spheres,
            attackables,
            interactables,
            nav_meshes,
            visibility,
            terrain_tags,
        ): Self::SystemData,
    ) {
        for event in input_events.read(&mut self.event_reader) {
            match *event {
                InputEvent::MouseButtonReleased(MouseButton::Left) => {
                    if !hide_cursor.hide {
                        if let Some(mouse_position) = input_handler.mouse_position() {
                            // Get the active camera if it is spawned and ready
                            let mut camera_join = (&cameras, &transforms).join();
                            if let Some((camera, camera_transform)) = active_camera
                                .entity
                                .and_then(|a| camera_join.get(a, &entities))
                                .or_else(|| camera_join.next())
                            {
                                let ray = camera.projection().screen_ray(
                                    Point2::new(mouse_position.0, mouse_position.1),
                                    Vector2::new(
                                        screen_dimensions.width(),
                                        screen_dimensions.height(),
                                    ),
                                    camera_transform,
                                );
                                let ncollide_ray = Ray::new(ray.origin, ray.direction);

                                let mut found_interactable = false;
                                let mut found_attackable = false;
                                let mut entity_id: Option<u32> = None;
                                let mut closest_distance = u32::max_value();
                                for (
                                    bounding_sphere,
                                    transform,
                                    entity,
                                    attackable,
                                    interactable,
                                ) in (
                                    &bounding_spheres,
                                    &transforms,
                                    &entities,
                                    (&attackables).maybe(),
                                    (&interactables).maybe(),
                                )
                                    .join()
                                {
                                    let visible =
                                        visibility.visible_unordered.contains(entity.id());
                                    let attackable = attackable.is_some();
                                    let interactable = interactable.is_some();
                                    if entity.id() != CURRENT_PLAYER
                                        && (interactable || attackable)
                                        && visible
                                    {
                                        let ball = Ball::new(bounding_sphere.radius);
                                        let ball_pos = transform.isometry();
                                        if ball.intersects_ray(ball_pos, &ncollide_ray) {
                                            let transform_distance = (camera_transform
                                                .translation()
                                                - transform.translation())
                                            .magnitude();
                                            let adjusted_distance: u32 = 0.0f32
                                                .max(transform_distance - bounding_sphere.radius)
                                                as u32;
                                            if adjusted_distance < closest_distance {
                                                closest_distance = adjusted_distance;
                                                found_interactable = interactable;
                                                found_attackable = attackable;
                                                entity_id = Some(entity.id());
                                            }
                                        }
                                    }
                                }

                                if let Some(entity_id) = entity_id {
                                    if found_attackable {
                                        attack_events.single_write(AttackEvent::new(
                                            CURRENT_PLAYER,
                                            entity_id,
                                            false,
                                        ));
                                    } else if found_interactable {
                                    }
                                } else {
                                    if let Some(position) = transforms
                                        .get(entities.entity(2))
                                        .and_then(|transform| Some(transform.translation()))
                                    {
                                        //test each mesh that is loaded
                                        for nav_mesh in nav_meshes.meshes_iter() {
                                            //TODO: clean this up, figure out how to map properly
                                            let mut vertices = vec![];
                                            for vertex in nav_mesh.vertices() {
                                                vertices.push(Point3::new(
                                                    vertex.x, vertex.y, vertex.z,
                                                ));
                                            }
                                            let mut indices = vec![];
                                            for triangle in nav_mesh.triangles() {
                                                indices.push(Point3::new(
                                                    triangle.first as usize,
                                                    triangle.second as usize,
                                                    triangle.third as usize,
                                                ));
                                            }
                                            let terrain_mesh =
                                                TriMesh::new(vertices, indices, None);
                                            let terrain_isometry = Isometry3::from_parts(
                                                Translation3::new(
                                                    nav_mesh.origin().x,
                                                    nav_mesh.origin().y,
                                                    nav_mesh.origin().z,
                                                ),
                                                *Transform::default().rotation(),
                                            );
                                            if let Some(toi) = terrain_mesh.toi_with_ray(
                                                &terrain_isometry,
                                                &ncollide_ray,
                                                true,
                                            ) {
                                                let mouse_world_position =
                                                    ncollide_ray.point_at(toi);

                                                let distance = (position
                                                    - Vector3::new(
                                                        mouse_world_position.x,
                                                        mouse_world_position.y,
                                                        mouse_world_position.z,
                                                    ))
                                                .magnitude();
                                                //maybe only have the rest of the pathfinding logic on backend
                                                if distance >= 1.0 {
                                                    if let Some(closest_nav_mesh) = nav_meshes
                                                        .closest_point(
                                                            (
                                                                mouse_world_position.x,
                                                                mouse_world_position.y,
                                                                mouse_world_position.z,
                                                            )
                                                                .into(),
                                                            NavQuery::Accuracy,
                                                        )
                                                    {
                                                        if let Some(path) = nav_meshes
                                                            .find_mesh(closest_nav_mesh.0)
                                                            .and_then(|mesh| {
                                                                mesh.find_path(
                                                                    (
                                                                        position.x, position.y,
                                                                        position.z,
                                                                    )
                                                                        .into(),
                                                                    closest_nav_mesh.1,
                                                                    NavQuery::Accuracy,
                                                                    NavPathMode::Accuracy,
                                                                )
                                                                .and_then(|path| {
                                                                    Some(
                                                                            path[1..]
                                                                                .iter()
                                                                                .map(|vec| {
                                                                                    Vector3::new(
                                                                                        vec.x,
                                                                                        vec.y,
                                                                                        vec.z,
                                                                                    )
                                                                                })
                                                                                .collect::<VecDeque<
                                                                                    Vector3<f32>,
                                                                                >>(
                                                                                ),
                                                                        )
                                                                })
                                                            })
                                                        {
                                                            movement_events.single_write(
                                                                MovementEvent::new(2, path),
                                                            );
                                                        } else {
                                                            error!("Failed to find path to ({}, {}, {}) for nav mesh {}", closest_nav_mesh.1.x, closest_nav_mesh.1.y, closest_nav_mesh.1.z, closest_nav_mesh.0.to_string());
                                                        }
                                                    } else {
                                                        error!("Failed to find nav mesh for point ({}, {}, {})", mouse_world_position.x, mouse_world_position.y, mouse_world_position.z)
                                                    }
                                                }

                                                //stop if intersection was found
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct TerrainTag;

#[derive(Clone, Serialize, Deserialize)]
struct ObstacleTag;

#[derive(Clone, Serialize, Deserialize)]
struct InteractableTag;

#[derive(Clone, Serialize, Deserialize)]
struct AttackableTag;

#[derive(Clone, Serialize, Deserialize, PrefabData)]
#[prefab(Component)]
struct AnimationComponent {
    idle_index: Option<usize>,
    run_index: Option<usize>,
    attack_index: Option<usize>,
}

impl Component for AnimationComponent {
    type Storage = DenseVecStorage<AnimationComponent>;
}

struct MovementEvent {
    source_entity: u32,
    path: VecDeque<Vector3<f32>>,
}

impl MovementEvent {
    fn new(source_entity: u32, path: VecDeque<Vector3<f32>>) -> MovementEvent {
        MovementEvent {
            source_entity,
            path: path,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PrefabData)]
#[prefab(Component)]
struct MovementComponent {
    //change to vec of points later
    movement_speed: f32,
    path: VecDeque<Vector3<f32>>,
}

impl Component for MovementComponent {
    type Storage = DenseVecStorage<MovementComponent>;
}
#[derive(SystemDesc)]
#[system_desc(name(MovementSystemDesc))]
struct MovementSystem {
    #[system_desc(event_channel_reader)]
    event_reader: ReaderId<MovementEvent>,
}

impl MovementSystem {
    fn new(event_reader: ReaderId<MovementEvent>) -> Self {
        MovementSystem {
            event_reader: event_reader,
        }
    }
}

impl<'a> System<'a> for MovementSystem {
    type SystemData = (
        Read<'a, EventChannel<MovementEvent>>,
        WriteStorage<'a, MovementComponent>,
        WriteStorage<'a, Transform>,
        Read<'a, Time>,
        Entities<'a>,
        WriteStorage<'a, AttackComponent>,
    );

    fn run(
        &mut self,
        (
            movement_events,
            mut movement_components,
            mut transforms,
            time,
            entities,
            mut attack_components,
        ): Self::SystemData,
    ) {
        for event in movement_events.read(&mut self.event_reader) {
            let entity = entities.entity(event.source_entity);
            if let Some(movement_component) = movement_components.get_mut(entity) {
                attack_components.remove(entity);
                movement_component.path = event.path.clone();
            }
        }

        for (transform, movement) in (&mut transforms, &mut movement_components).join() {
            if let Some(move_position) = movement.path.iter().next() {
                let distance = (transform.translation() - move_position).magnitude();
                let distance_to_move = movement.movement_speed * time.delta_seconds();

                if distance != 0.0 && distance >= distance_to_move {
                    transform.face_towards(*move_position, Vector3::new(0.0, 1.0, 0.0));
                    transform.move_forward(distance_to_move);
                } else {
                    movement.path.pop_front();
                }
            }
        }
    }
}

#[derive(SystemDesc)]
#[system_desc(name(AnimationSystemDesc))]
struct AnimationSystem;

impl<'a> System<'a> for AnimationSystem {
    type SystemData = (
        ReadStorage<'a, AnimationSet<usize, Transform>>,
        WriteStorage<'a, AnimationControlSet<usize, Transform>>,
        ReadStorage<'a, MovementComponent>,
        ReadStorage<'a, AnimationComponent>,
        Entities<'a>,
        ReadStorage<'a, AttackComponent>,
    );

    fn run(
        &mut self,
        (
            animation_sets,
            mut animation_control_sets,
            movement_components,
            animation_components,
            entities,
            attack_components,
        ): Self::SystemData,
    ) {
        for (animation_component, entity) in (&animation_components, &entities).join() {
            if let Some((entity, Some(animations))) =
                Some(entity).map(|entity| (entity, animation_sets.get(entity)))
            {
                let set =
                    get_animation_set::<usize, Transform>(&mut animation_control_sets, entity)
                        .unwrap();
                let running = movement_components
                    .get(entity)
                    .and_then(|movement| movement.path.iter().next())
                    .is_some();
                let attack_component = attack_components.get(entity);
                let attacking = match attack_component {
                    Some(attack) => {
                        attack.attacking
                            || match animation_component.attack_index {
                                Some(attack_index) => set.has_animation(attack_index),
                                _ => false,
                            }
                    }
                    _ => false,
                };

                if running {
                    if let Some(animation_index) = animation_component.run_index {
                        let animation = animations.animations.get(&animation_index).unwrap();
                        if !set.has_animation(animation_index) {
                            if let Some(previous_animation) = set.animations.last() {
                                set.abort(previous_animation.0);
                            }
                            set.add_animation(
                                animation_index,
                                animation,
                                EndControl::Stay,
                                1.0,
                                AnimationCommand::Start,
                            );
                        }
                    }
                } else if attacking {
                    if let Some(animation_index) = animation_component.attack_index {
                        let animation = animations.animations.get(&animation_index).unwrap();
                        if !set.has_animation(animation_index) {
                            if let Some(previous_animation) = set.animations.last() {
                                set.abort(previous_animation.0);
                            }
                            set.add_animation(
                                animation_index,
                                animation,
                                EndControl::Stay,
                                1.0,
                                AnimationCommand::Start,
                            );
                        }
                    }
                } else {
                    if let Some(animation_index) = animation_component.idle_index {
                        let animation = animations.animations.get(&animation_index).unwrap();
                        if !set.has_animation(animation_index) {
                            if let Some(previous_animation) = set.animations.last() {
                                set.abort(previous_animation.0);
                            }
                            set.add_animation(
                                animation_index,
                                animation,
                                EndControl::Stay,
                                1.0,
                                AnimationCommand::Start,
                            );
                        }
                    }
                }
            }
        }
    }
}

struct AttackEvent {
    //entity ids which will be generated by server
    source_entity: u32,
    target_entity: u32,
    retaliation: bool,
}

impl AttackEvent {
    fn new(source_entity: u32, target_entity: u32, retaliation: bool) -> AttackEvent {
        AttackEvent {
            source_entity,
            target_entity,
            retaliation,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PrefabData)]
#[prefab(Component)]
struct AttackComponent {
    target_entity: u32,
    cooldown: f32,
    attacking: bool,
}

impl Component for AttackComponent {
    type Storage = DenseVecStorage<AttackComponent>;
}

#[derive(SystemDesc)]
#[system_desc(name(AttackSystemDesc))]
struct AttackSystem {
    #[system_desc(event_channel_reader)]
    attack_event_reader: ReaderId<AttackEvent>,
}

impl AttackSystem {
    fn new(event_reader: ReaderId<AttackEvent>) -> Self {
        AttackSystem {
            attack_event_reader: event_reader,
        }
    }
}

//most of this logic will probably move to server side... client will just receive damage events and start/stop attacking events for animations
impl<'a> System<'a> for AttackSystem {
    type SystemData = (
        Write<'a, EventChannel<AttackEvent>>,
        WriteStorage<'a, AttackComponent>,
        WriteStorage<'a, MovementComponent>,
        WriteStorage<'a, Transform>,
        Read<'a, Time>,
        Entities<'a>,
        Read<'a, NavMeshesRes>,
    );

    //process attack events and for each event add attack components with target_entity from event
    //if entity has attack component
    //  get attack source_entity and target_entity positions
    //  if target_entity close enough to source_entity
    //      face toward target_entity_position
    //      set attack component's attacking bool to true
    //      set movement component's move_position to None
    //      publish damage event?
    //  else
    //      set attack component's attacking bool to false
    //      set movement component's move_position to Some(target_entity_position)
    fn run(
        &mut self,
        (
            mut attack_events,
            mut attack_components,
            mut movement_components,
            mut transforms,
            time,
            entities,
            nav_meshes,
        ): Self::SystemData,
    ) {
        for event in attack_events.read(&mut self.attack_event_reader) {
            if event.source_entity != event.target_entity {
                let entity = entities.entity(event.source_entity);
                let should_attack = match attack_components.get(entity) {
                    Some(attack) => {
                        !event.retaliation && attack.target_entity != event.target_entity
                    }
                    None => true,
                };

                if should_attack {
                    attack_components.insert(
                        entity,
                        AttackComponent {
                            target_entity: event.target_entity,
                            cooldown: ATTACK_INTERVAL,
                            attacking: false,
                        },
                    );
                }
            }
        }

        for (source_transform, attack_component, movement_component, source_entity) in (
            &transforms,
            &mut attack_components,
            &mut movement_components,
            &entities,
        )
            .join()
        {
            if let Some(target_transform) =
                transforms.get(entities.entity(attack_component.target_entity))
            {
                let source_position = source_transform.translation();
                let target_position = target_transform.translation();
                let attack_distance = 3.0; //this value should be roughly the sum of the radii of the bounding spheres from both entities + weapon distance
                let source_direction = (source_position - target_position)
                    / (source_position - target_position).magnitude();
                let move_position =
                    target_transform.translation() + attack_distance * source_direction;
                let delta_time = time.delta_seconds();
                let distance = (source_position - move_position).magnitude();
                let move_distance = movement_component.movement_speed * delta_time;

                if distance == 0.0 || distance < move_distance {
                    movement_component.path.clear();
                    attack_component.cooldown -= delta_time;
                    if attack_component.cooldown <= 0.0 {
                        //send "attacking" message from server to client here
                        attack_component.cooldown = ATTACK_INTERVAL;
                        attack_component.attacking = true;

                        let target_entity = entities.entity(attack_component.target_entity);
                        attack_events.single_write(AttackEvent::new(
                            target_entity.id(),
                            source_entity.id(),
                            true,
                        ));
                    } else {
                        //probably not safe to do it this way since animation system might not see it before it changes again
                        attack_component.attacking = false;
                    }

                // source_transform.face_towards(
                //     Vector3::new(target_position.x, target_position.y, target_position.z),
                //     Vector3::new(0.0, 1.0, 0.0),
                // );
                } else {
                    attack_component.attacking = false;
                    //find path
                    if let Some(closest_nav_mesh) = nav_meshes.closest_point(
                        (move_position.x, move_position.y, move_position.z).into(),
                        NavQuery::Accuracy,
                    ) {
                        if let Some(path) =
                            nav_meshes.find_mesh(closest_nav_mesh.0).and_then(|mesh| {
                                mesh.find_path(
                                    (source_position.x, source_position.y, source_position.z)
                                        .into(),
                                    closest_nav_mesh.1,
                                    NavQuery::Accuracy,
                                    NavPathMode::Accuracy,
                                )
                                .and_then(|path| {
                                    Some(
                                        path[1..]
                                            .iter()
                                            .map(|vec| Vector3::new(vec.x, vec.y, vec.z))
                                            .collect::<VecDeque<Vector3<f32>>>(),
                                    )
                                })
                            })
                        {
                            movement_component.path = path;
                        }
                    }
                }
            }
        }
    }
}

//maybe separate out entity stats into new component with movement_speed, attack_speed, etc

//generate navmesh from terrain and walkable meshes like bridges and then cut out obstacles
