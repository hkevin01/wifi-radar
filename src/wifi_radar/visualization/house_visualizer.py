"""
ID: WR-VIZ-HOUSE-001
Requirement: Provide an optional real-time 2-D top-down / perspective house view
             rendered with pygame, overlaying detected person positions on a
             schematic floor plan.
Purpose: Gives operators a spatial, room-level overview of where people are
         located inside the monitored area, complementing the 3-D skeleton view
         in the Dash dashboard.
Thread-safety model:
    ``update_people()`` is the only method intended to be called from the
    processing thread.  All shared state (``self._people``) is protected by
    ``self._lock`` (threading.Lock).  The render loop reads ``self._people``
    under the same lock, ensuring no partial writes are visible.
Assumptions:
    - pygame must be installed; if not, the visualiser degrades gracefully and
      all public methods become no-ops.
    - A display server (X11, Wayland, or headless framebuffer) must be available.
Failure Modes:
    Any pygame or OpenGL exception in ``_render_loop()`` is caught, logged, and
    causes the loop to exit cleanly without crashing the main process.
"""
from __future__ import annotations

import logging
import threading
from typing import List


class HouseVisualizer:
    """Renders a top-down or perspective 3-D house view with pose overlays.

    ID: WR-VIZ-HOUSE-CLASS-001
    Requirement: Display detected person positions on a 2-D top-down floor plan
                 rendered with pygame at a configured FPS.
    Purpose: Give operators a room-level spatial overview of occupant locations
             to complement the 3-D skeleton view in the Dash dashboard.
    Rationale: Running the render loop in a daemon thread decouples the display
               update rate from the inference pipeline throughput.
    Inputs:
        width, height — int: pygame window dimensions in pixels.
        fps — int: target render frame rate.
        wall_transparency — float [0,1]: alpha for floor-plan wall overlays.
    Outputs:
        Live pygame window with person position overlays.
    Preconditions:
        pygame must be installed and a display server must be available.
    Assumptions:
        Keypoint coordinates are normalised to [-1, 1] on each axis.
    Constraints:
        Degrades gracefully (no-op all methods) if pygame is not installed.
    References:
        pygame documentation; WR-VIZ-HOUSE-001 module docstring.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        fps: int = 30,
        wall_transparency: float = 0.5,
    ) -> None:
        """Initialise window parameters and attempt a lazy pygame import.

        ID: WR-VIZ-HOUSE-INIT-001
        Requirement: Store configuration, initialise threading primitives, and
                     attempt a lazy import of pygame; set self._pygame_available
                     accordingly.
        Purpose: Allow HouseVisualizer to be constructed even on systems without
                 a display, so the rest of the pipeline can run headlessly.
        Rationale: Lazy import inside __init__ rather than module level lets
                   tests and non-display environments import the class safely.
        Inputs:
            width              — int > 0: pygame window width in pixels.
            height             — int > 0: pygame window height in pixels.
            fps                — int > 0: target render frame rate.
            wall_transparency  — float [0, 1]: wall overlay alpha.
        Outputs:
            None — initialises self.
        Preconditions:
            None — no display or network interaction.
        Postconditions:
            self._pygame_available == True iff pygame is importable.
            self._running == False; self._people == [].
        Assumptions:
            Caller will call start() before the render loop is needed.
        Side Effects:
            Attempts `import pygame`; logs WARNING if unavailable.
        Failure Modes:
            pygame not installed: ImportError caught; self._pygame_available=False.
        Error Handling:
            ImportError silently handled; warning logged.
        Constraints:
            No threads or subprocesses are started at construction time.
        Verification:
            Unit test: construct without pygame installed; assert _pygame_available==False.
        References:
            pygame.init(); threading.Lock; WR-VIZ-HOUSE-CLASS-001.
        """
        self.logger = logging.getLogger("HouseVisualizer")
        self.width = width
        self.height = height
        self.fps = fps
        self.wall_transparency = wall_transparency

        self._running = False
        self._thread: threading.Thread | None = None
        self._people: List[dict] = []
        self._lock = threading.Lock()

        # Lazy-import pygame so the rest of the system works without a display.
        try:
            import pygame  # noqa: F401
            self._pygame_available = True
        except ImportError:
            self.logger.warning(
                "pygame is not installed — house visualizer disabled. "
                "Install it with:  pip install pygame"
            )
            self._pygame_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pygame render loop in a background daemon thread.

        ID: WR-VIZ-HOUSE-START-001
        Requirement: Set self._running=True and spawn a daemon thread targeting
                     _render_loop() when pygame is available and not already running.
        Purpose: Activate the display so operators see real-time position updates.
        Rationale: Daemon thread ensures automatic cleanup when the main process exits
                   without requiring an explicit stop() call.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            self._pygame_available == True.
            self._running == False.
        Postconditions:
            self._running == True; self._thread is a live daemon thread.
        Assumptions:
            A display server (X11/Wayland/headless FB) is available.
        Side Effects:
            Sets self._running = True.
            Spawns self._thread as a daemon thread.
            Logs INFO message.
        Failure Modes:
            pygame not available: method returns immediately (no-op).
        Error Handling:
            Duplicate start(): returns immediately (no-op guard).
        Constraints:
            Only one render thread should be active at a time.
        Verification:
            Integration test: start(); sleep(0.1); assert _thread.is_alive().
        References:
            threading.Thread daemon=True; _render_loop().
        """
        if not self._pygame_available:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self.logger.info("HouseVisualizer started (%dx%d @ %d fps)", self.width, self.height, self.fps)

    def stop(self) -> None:
        """Signal the render loop to exit and wait for the thread to join.

        ID: WR-VIZ-HOUSE-STOP-001
        Requirement: Set self._running=False and join the render thread within
                     a 2-second timeout.
        Purpose: Ensure pygame is cleanly quit and the window is closed without
                 leaving an orphaned thread after the pipeline shuts down.
        Rationale: A 2-second timeout prevents indefinite blocking if pygame
                   hangs on quit(); the daemon flag ensures forced cleanup on exit.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            start() must have been called (self._thread must exist).
        Postconditions:
            self._running == False; render thread has exited or timed out.
        Assumptions:
            pygame.quit() completes within 2 seconds under normal operation.
        Side Effects:
            Sets self._running = False.
            Joins self._thread with timeout=2.0 s.
            Logs INFO message.
        Failure Modes:
            Thread hangs: join() times out after 2 s; thread becomes abandoned.
        Error Handling:
            None beyond the 2-second timeout.
        Constraints:
            Thread cleanup bounded to 2 s maximum wait.
        Verification:
            Integration test: start(); stop(); assert not _thread.is_alive() (within 2s).
        References:
            threading.Thread.join(timeout); pygame.quit().
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("HouseVisualizer stopped")

    def update_people(self, people: List[dict]) -> None:
        """Thread-safe replacement of the detected-people list for the next frame.

        ID: WR-VIZ-HOUSE-UPDATE-001
        Requirement: Replace self._people with the provided list under self._lock
                     so the render thread always sees a consistent snapshot.
        Purpose: Allow the inference thread to publish updated occupant positions
                 without blocking on the render thread.
        Rationale: Holding self._lock only during the list assignment (not during
                   rendering) minimises lock contention between producer and consumer.
        Inputs:
            people — List[dict]: each dict must contain:
                'keypoints'  — (17,3) numpy float array, normalised [-1,1].
                'confidence' — (17,) numpy float array, per-keypoint scores [0,1].
        Outputs:
            None — updates self._people as a side effect.
        Preconditions:
            None — safe to call before start().
        Postconditions:
            self._people is a new list containing the provided items.
        Assumptions:
            Render thread reads self._people under the same self._lock.
        Side Effects:
            Acquires self._lock.
            Replaces self._people with list(people).
        Failure Modes:
            Invalid dict entries: render loop skips entries missing required keys.
        Error Handling:
            None at this level; defensive checks are in _draw_people.
        Constraints:
            Lock held only for the duration of list(people) and assignment.
        Verification:
            Unit test: set people=[{...}]; assert self._people == [{...}].
        References:
            threading.Lock; _draw_people(); WR-VIZ-HOUSE-CLASS-001.
        """
        with self._lock:
            self._people = list(people)

    # ------------------------------------------------------------------
    # Internal rendering loop (placeholder — extend with real pygame/OpenGL)
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        """Pygame event-and-render loop running in the background daemon thread.

        ID: WR-VIZ-HOUSE-LOOP-001
        Requirement: Initialise pygame, run an event-pump+draw loop at self.fps,
                     and call pygame.quit() in a finally block on exit.
        Purpose: Maintain the live display window and render person positions at
                 the configured frame rate until self._running is set False.
        Rationale: The event pump (pygame.event.get) must run in the same thread
                   that called pygame.init() to avoid display server race conditions.
        Inputs:
            None — reads self._running, self._people (via _draw_people), self.fps.
        Outputs:
            None — renders to the pygame window as a side effect.
        Preconditions:
            Called from start() as a daemon thread.
            pygame must be installed.
        Postconditions:
            pygame.quit() is called in the finally block regardless of exit reason.
        Assumptions:
            A display server is available and pygame.init() succeeds.
        Side Effects:
            Calls pygame.init() and pygame.display.set_mode().
            Calls pygame.quit() in finally block.
            Calls _draw_people() under self._lock on each iteration.
        Failure Modes:
            pygame.init() fails: exception caught, logged; pygame.quit() called.
            Any render error: caught, logged; loop exits.
        Error Handling:
            Top-level except catches all exceptions; logs error; pygame.quit() called.
        Constraints:
            Frame rate capped by clock.tick(self.fps).
        Verification:
            Integration test: start(); sleep(0.5); assert window title matches config.
        References:
            pygame.event.get; pygame.display.flip; clock.tick.
        """
        import time
        import pygame

        try:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("WiFi-Radar — House View")
            clock = pygame.time.Clock()

            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False

                screen.fill((20, 20, 30))
                with self._lock:
                    self._draw_people(screen)

                pygame.display.flip()
                clock.tick(self.fps)

        except Exception as exc:
            self.logger.error("Render loop error: %s", exc)
        finally:
            pygame.quit()

    def _draw_people(self, screen) -> None:  # type: ignore[no-untyped-def]
        """Render each detected person as a circle at their projected 2-D position.

        ID: WR-VIZ-HOUSE-DRAW-001
        Requirement: For each person in self._people with at least one keypoint
                     confidence > 0.3, compute a centre-of-mass screen position
                     and render a filled circle at that location.
        Purpose: Provide a simple but informative room-level view of occupant
                 positions without requiring a full 3-D rendering pipeline.
        Rationale: Mean of high-confidence keypoints is a robust centroid estimate
                   that degrades gracefully when only a subset of keypoints is visible.
        Inputs:
            screen — pygame.Surface: the active display window surface.
        Outputs:
            None — draws to screen as a side effect.
        Preconditions:
            Must be called from inside `with self._lock` block in _render_loop().
            pygame must be initialised.
        Postconditions:
            One filled circle per valid person is drawn on screen.
        Assumptions:
            Keypoint X and Y are normalised to [-1, 1].
            pygame.draw.circle is available.
        Side Effects:
            Calls pygame.draw.circle for each valid person.
        Failure Modes:
            Person dict missing 'keypoints' or 'confidence': skipped.
            No valid keypoints (all conf <= 0.3): person skipped.
        Error Handling:
            Defensive .get() checks for missing keys; valid.any() guard.
        Constraints:
            Called from the render loop; must be fast (<< 1/fps seconds).
        Verification:
            Unit test: mock screen; call with synthetic people; assert circle called.
        References:
            pygame.draw.circle; COCO-17 keypoints; normalised coordinate system.
        """
        import pygame

        for person in self._people:
            kp = person.get("keypoints")
            conf = person.get("confidence")
            if kp is None or conf is None:
                continue
            # Centre of mass — average of high-confidence keypoints
            valid = conf > 0.3
            if not valid.any():
                continue
            cx = float(kp[valid, 0].mean())
            cy = float(kp[valid, 1].mean())
            sx = int((cx + 1) / 2 * self.width)
            sy = int((cy + 1) / 2 * self.height)
            pygame.draw.circle(screen, (0, 200, 255), (sx, sy), 12)
