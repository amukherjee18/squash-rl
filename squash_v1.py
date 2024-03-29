import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pong")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define constants
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
BALL_SIZE = 20
WALL_WIDTH = 10
PADDLE_SPEED = 5
BALL_SPEED = 5

# Define fonts
font = pygame.font.Font(None, 36)

# Define initial positions
player_score = 0
opponent_score = 0
player_y = (SCREEN_HEIGHT - PADDLE_HEIGHT) // 2
opponent_y = (SCREEN_HEIGHT - PADDLE_HEIGHT) // 2
ball_x = SCREEN_WIDTH // 2 - 20
ball_y = SCREEN_HEIGHT // 2
ball_dx = BALL_SPEED * random.choice([-1, 1])
ball_dy = BALL_SPEED * random.choice([-1, 1])

# Function to draw paddles, ball, and wall
def draw_objects():
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (0, player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH - PADDLE_WIDTH, opponent_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, ((SCREEN_WIDTH - WALL_WIDTH) // 2, 0, WALL_WIDTH, SCREEN_HEIGHT))
    pygame.draw.ellipse(screen, WHITE, (ball_x - BALL_SIZE // 2, ball_y - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE))

# Function to handle collisions
def check_collision():
    global ball_x, ball_y, ball_dx, ball_dy, player_score, opponent_score
    # Ball collision with top/bottom walls
    if ball_y - BALL_SIZE // 2 <= 0 or ball_y + BALL_SIZE // 2 >= SCREEN_HEIGHT:
        ball_dy = -ball_dy

    # Ball collision with paddles
    if ball_x - BALL_SIZE // 2 <= PADDLE_WIDTH and player_y < ball_y < player_y + PADDLE_HEIGHT:
        ball_dx = -ball_dx
    elif ball_x + BALL_SIZE // 2 >= SCREEN_WIDTH - PADDLE_WIDTH and opponent_y < ball_y < opponent_y + PADDLE_HEIGHT:
        ball_dx = -ball_dx

    # Ball collision with the wall
    if (SCREEN_WIDTH - WALL_WIDTH) // 2 <= ball_x + BALL_SIZE // 2 <= (SCREEN_WIDTH + WALL_WIDTH) // 2:
        ball_dx = -ball_dx

    # Ball out of bounds
    if ball_x - BALL_SIZE // 2 <= 0:
        opponent_score += 1
        reset_ball()
    elif ball_x + BALL_SIZE // 2 >= SCREEN_WIDTH:
        player_score += 1
        reset_ball()

# Function to reset the ball
def reset_ball():
    global ball_x, ball_y, ball_dx, ball_dy
    ball_x = SCREEN_WIDTH // 2 - 20
    ball_y = SCREEN_HEIGHT // 2
    ball_dx = BALL_SPEED * random.choice([-1, 1])
    ball_dy = BALL_SPEED * random.choice([-1, 1])

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player_y > 0:
        player_y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and player_y < SCREEN_HEIGHT - PADDLE_HEIGHT:
        player_y += PADDLE_SPEED

    # Simple opponent AI
    if ball_dy < 0 and opponent_y > 0:
        opponent_y -= PADDLE_SPEED
    if ball_dy > 0 and opponent_y < SCREEN_HEIGHT - PADDLE_HEIGHT:
        opponent_y += PADDLE_SPEED

    # Update ball position
    ball_x += ball_dx
    ball_y += ball_dy

    # Check collisions
    check_collision()

    # Draw everything
    draw_objects()

    # Draw scores
    player_text = font.render("Player: " + str(player_score), True, WHITE)
    opponent_text = font.render("Opponent: " + str(opponent_score), True, WHITE)
    screen.blit(player_text, (50, 50))
    screen.blit(opponent_text, (SCREEN_WIDTH - 250, 50))

    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
