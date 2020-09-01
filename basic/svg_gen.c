
/* Copyright 2020, Michael Serra
   All rights reserved. */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include <SDL/SDL_gfxPrimitives.h>

#define NUM_TRIGONS		600

#define MIN_ALPHA		15
#define MAX_ALPHA		80

#define MIN_AREA		1000
#define MAX_AREA		3600
#define MAX_BIG_AREA	12000
#define BIG_ODDS		20

#define CHANNEL_DELTA	32
#define RGB_DELTA		16
#define ALPHA_DELTA		12

#define SAMPLING_RATE	1
#define SAMPLING_DRIFT	24

#define X_SCATTER		10
#define Y_SCATTER		9

#define BG_R			210
#define BG_G			180
#define BG_B			140

//----------------------------------------------------

#define MAX(a, b)	(((a) > (b)) ? (a) : (b))

#define SIGN(n)		((n) * ((rand() % 2) ? 1 : -1))

#define CLAMP(m, n)	((m) < 0 ? 0 : ((m) > n ? n : (m)))
#define BYTE(n)		((n) < 0 ? 0 : ((n) > 255 ? 255 : (n)))

#define getR(c)		((c)  & 0xFF)
#define getG(c)		(((c) & 0xFF00) >> 8)
#define getB(c)		(((c) & 0xFF0000) >> 16)
#define setR(c, v)	(((c) & 0xFFFF00) + (v))
#define setG(c, v)	(((c) & 0xFF00FF) + ((v) << 8))
#define setB(c, v)	(((c) & 0x00FFFF) + ((v) << 16))

//----------------------------------------------------

typedef struct
{
	int32_t x, y;
} point;

typedef struct
{
	int32_t r, g, b, a;
} rgba;

typedef struct
{
	point p, q;
} rect;

typedef struct
{
	point a, b, c;
	rgba p;
} tri;

//------------------------------------------------------

int32_t xres, yres;

rgba sample[NUM_TRIGONS];

SDL_Surface* image;
SDL_Surface* screen;

//------------------------------------------------------

void init(int argc, char* filename)
{
	const int flags = SDL_SWSURFACE | SDL_DOUBLEBUF;

	if (argc < 2)
	{
		printf("A path to an image file is required.\n");
		exit(0);
	}
	image  = IMG_Load(filename);
	xres   = image->w;
	yres   = image->h;
	screen = SDL_SetVideoMode(xres, yres, 0, flags);
}

uint32_t getPixel(SDL_Surface *s, int x, int y)
{
	int bpp  = s->format->BytesPerPixel;
	uint8_t* p = (uint8_t *) s->pixels + y * s->pitch + x * bpp;

	switch (bpp)
	{
		case 1:
			return *p;
		case 2:
			return *(uint16_t *)p;
		case 3:
			if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
				return p[0] << 16 | p[1] << 8 | p[2];
			else
				return p[0] | p[1] << 8 | p[2] << 16;
		case 4:
			return *(uint32_t *) p;
		default:
			return 0;   // shouldn't happen; avoids warnings
	}
}

void copyInputImage(uint32_t img[][yres])
{
	// We transfer the input image surface to an array
	// because SDL_Surface has slow direct pixel access.

	for (int32_t x = 0; x < xres; x++)
	{
		for (int32_t y = 0; y < yres; y++)
		{
			img[x][y] = getPixel(image, x, y);
		}
	}
	SDL_FreeSurface(image);
}

void sample_img(uint32_t img[][yres])
{
	int i = 0;

	for (; i < NUM_TRIGONS / SAMPLING_RATE; i++)
	{
		rgba c;
		uint32_t p = img[rand() % xres][rand() % yres];

		c.r = BYTE(((int32_t) getR(p)) + SIGN(SAMPLING_DRIFT));
		c.g = BYTE(((int32_t) getG(p)) + SIGN(SAMPLING_DRIFT));
		c.b = BYTE(((int32_t) getB(p)) + SIGN(SAMPLING_DRIFT));

		c.a = MIN_ALPHA + (rand() % MAX_ALPHA);

		sample[i] = c;
	}
	for (; i < NUM_TRIGONS; i++)
	{
		rgba p;

		p.r = rand() % 256;
		p.g = rand() % 256;
		p.b = rand() % 256;
		p.a = MIN_ALPHA + (rand() % MAX_ALPHA);

		sample[i] = p;
	}
}

//------------------------------------------------------

rgba copy_rgba(rgba p)
{
	rgba q;

	q.r = p.r;
	q.g = p.g;
	q.b = p.b;
	q.a = p.a;

	return q;
}

tri copy_trigon(tri t)
{
	tri u;

	u.a.x = t.a.x;
	u.a.y = t.a.y;
	u.b.x = t.b.x;
	u.b.y = t.b.y;
	u.c.x = t.c.x;
	u.c.y = t.c.y;
	u.p.r = t.p.r;
	u.p.g = t.p.g;
	u.p.b = t.p.b;
	u.p.a = t.p.a;

	return u;
}

uint64_t zpow(uint32_t base, uint32_t exp)
{
    uint64_t n = 1;

	while (exp)
	{
		if (exp & 1)
		{
			n *= base;
		}
		exp >>= 1;
		base *= base;
	}
    return n;
}

uint32_t dist(point p, point q)
{
	return zpow(q.x - p.x, 2) + zpow(q.y - p.y, 2);
}

uint32_t area(tri t)
{
	uint32_t n =
		(t.b.x - t.a.x) * 
		(t.c.y - t.a.y) -
		(t.b.y - t.a.y) *
		(t.c.x - t.a.x);

	return abs(n);
}

bool rotund(tri t)
{
	uint32_t l = dist(t.a, t.b);
	uint32_t m = dist(t.b, t.c);
	uint32_t n = dist(t.c, t.a);

	uint32_t a = abs(l - m);
	uint32_t b = abs(l - n);
	uint32_t c = abs(m - n);

	return MAX(MAX(a, b), c) < area(t) * 2;
}

point centroid(tri t)
{
	uint32_t sumX = 0;
	uint32_t sumY = 0;

	sumX += t.a.x;
	sumX += t.b.x;
	sumX += t.c.x;
	sumY += t.a.y;
	sumY += t.b.y;
	sumY += t.c.y;

	point p;

	p.x = sumX / 3;
	p.y = sumY / 3;

	return p;
}

tri scale(tri t, int n)
{
	point p = centroid(t);

	tri u = copy_trigon(t);

	u.a.x -= p.x;
	u.b.x -= p.x;  // place triangle
	u.c.x -= p.x;  // center of mass
	u.a.y -= p.y;  // (its centroid)
	u.b.y -= p.y;  // at the origin
	u.c.y -= p.y;

	u.a.x *= n;
	u.b.x *= n;
	u.c.x *= n;
	u.a.y *= n;
	u.b.y *= n;
	u.c.y *= n;

	u.a.x += (n * p.x);
	u.b.x += (n * p.x);  // restore triangle
	u.c.x += (n * p.x);  // to its scaled
	u.a.y += (n * p.y);  // original position
	u.b.y += (n * p.y);
	u.c.y += (n * p.y);

	return u;
}

tri triangle()  // rgb from sample; random alpha
{
	tri t;

	t.p   = copy_rgba(sample[rand() % NUM_TRIGONS]);
	t.p.a = MIN_ALPHA + (rand() % MAX_ALPHA);

	t.a.x = rand() % xres;
	t.a.y = rand() % yres;
	t.b.x = rand() % xres;
	t.b.y = rand() % yres;
	t.c.x = rand() % xres;
	t.c.y = rand() % yres;

	uint32_t size = area(t);

	if (rand() % BIG_ODDS == 0)  // big triangle
	{
		while (size > MAX_BIG_AREA || !rotund(t))
		{
			t.a.x = rand() % xres;
			t.a.y = rand() % yres;
			t.b.x = rand() % xres;
			t.b.y = rand() % yres;
			t.c.x = rand() % xres;
			t.c.y = rand() % yres;
			size  = area(t);
		}
	}
	else  // small triangle
	{
		while (MIN_AREA < size || size > MAX_AREA || !rotund(t))
		{
			t.a.x = rand() % xres;
			t.a.y = rand() % yres;
			t.b.x = rand() % xres;
			t.b.y = rand() % yres;
			t.c.x = rand() % xres;
			t.c.y = rand() % yres;
			size  = area(t);
		}
	}

	// scatter triangles a bit away from center
	t.a.x = t.a.x < xres / 2 ? t.a.x - X_SCATTER : t.a.x + X_SCATTER;
	t.a.y = t.a.y < yres / 2 ? t.a.y - Y_SCATTER : t.a.y + Y_SCATTER;
	t.b.x = t.b.x < xres / 2 ? t.b.x - X_SCATTER : t.b.x + X_SCATTER;
	t.b.y = t.b.y < yres / 2 ? t.b.y - Y_SCATTER : t.b.y + Y_SCATTER;
	t.c.x = t.c.x < xres / 2 ? t.c.x - X_SCATTER : t.c.x + X_SCATTER;
	t.c.y = t.c.y < yres / 2 ? t.c.y - Y_SCATTER : t.c.y + Y_SCATTER;

	return t;
}

uint64_t rgb_dist(int r, int g, int b, int r2, int g2, int b2)
{
	__m128i intV0 = _mm_set_epi32(r,  g,  b,  0);
	__m128i intV1 = _mm_set_epi32(r2, g2, b2, 0);

	__m128i diffs = _mm_sub_epi32(intV0, intV1);

	__m128i squares = _mm_mul_epi32(diffs, diffs);
			squares = _mm_abs_epi32(diffs);

	int* ns = (int*) &squares;

	return (uint64_t) ns[0] + ns[1] + ns[2] + ns[3];
}

int compare(const void *p, const void *q)
{
	int x = *(const int *) p;
	int y = *(const int *) q;

	return x < y ? -1 : (x > y ? 1 : 0);
}

void sort(int *arr, size_t n)
{
	qsort(arr, n, sizeof(int), compare);
}

rect tri_bounds(tri t)
{
	int xs[3];
	int ys[3];

	xs[0] = CLAMP(t.a.x, xres);
	xs[1] = CLAMP(t.b.x, xres);
	xs[2] = CLAMP(t.c.x, xres);
	ys[0] = CLAMP(t.a.y, yres);
	ys[1] = CLAMP(t.b.y, yres);
	ys[2] = CLAMP(t.c.y, yres);

	sort(xs, 3);
	sort(ys, 3);

	rect r;

	r.p.x = xs[0];
	r.p.y = ys[0];
	r.q.x = xs[2];
	r.q.y = ys[2];

	return r;
}

rect bounds(tri t, tri u)
{
	int xs[6];
	int ys[6];

	xs[0] = CLAMP(t.a.x, xres);
	xs[1] = CLAMP(t.b.x, xres);
	xs[2] = CLAMP(t.c.x, xres);
	xs[3] = CLAMP(u.a.x, xres);
	xs[4] = CLAMP(u.b.x, xres);
	xs[5] = CLAMP(u.c.x, xres);

	ys[0] = CLAMP(t.a.y, yres);
	ys[1] = CLAMP(t.b.y, yres);
	ys[2] = CLAMP(t.c.y, yres);
	ys[3] = CLAMP(u.a.y, yres);
	ys[4] = CLAMP(u.b.y, yres);
	ys[5] = CLAMP(u.c.y, yres);

	sort(xs, 6);
	sort(ys, 6);

	rect r;

	r.p.x = xs[0];
	r.p.y = ys[0];
	r.q.x = xs[5];
	r.q.y = ys[5];

	return r;
}

//----------------------------------------------------------------

int64_t tri_cost(SDL_Surface* s, uint32_t img[][yres], tri t)
{
	uint8_t r, g, b;
	int r2, g2, b2;

	const rect pq = tri_bounds(t);

	uint64_t n = 0;

	for (int32_t x = pq.p.x; x <= pq.q.x; x++)
	{
		for (int32_t y = pq.p.y; y <= pq.q.y; y++)
		{
			SDL_GetRGB(getPixel(s, x, y), s->format, &r, &g, &b);

			r2 = getR(img[x][y]);
			g2 = getG(img[x][y]);
			b2 = getB(img[x][y]);

			n += rgb_dist(r, g, b, r2, g2, b2);
		}
	}
	return n;
}

int64_t cost(SDL_Surface* s, uint32_t img[][yres], tri t, tri u)
{
	uint8_t r, g, b;
	int r2, g2, b2;

	const rect pq = bounds(t, u);

	uint64_t n = 0;

	for (int32_t x = pq.p.x; x <= pq.q.x; x++)
	{
		for (int32_t y = pq.p.y; y <= pq.q.y; y++)
		{
			SDL_GetRGB(getPixel(s, x, y), s->format, &r, &g, &b);

			r2 = getR(img[x][y]);
			g2 = getG(img[x][y]);
			b2 = getB(img[x][y]);

			n += rgb_dist(r, g, b, r2, g2, b2);
		}
	}
	return n;
}

//----------------------------------------------------------------

void draw(SDL_Surface* s, tri* trigons)
{
	tri t;
	SDL_FillRect(s, NULL, SDL_MapRGB(s->format, BG_R, BG_G, BG_B));

	for (int i = 0; i < NUM_TRIGONS; i++)
	{
		t = trigons[i];

		filledTrigonRGBA(s, t.a.x, t.a.y, t.b.x, t.b.y,
			t.c.x, t.c.y, t.p.r, t.p.g, t.p.b, t.p.a);
	}
}

void writeScaledBMP(tri* trigons, uint32_t n)
{
	tri ts[NUM_TRIGONS];

	for (int i = 0; i < NUM_TRIGONS; i++)
	{
		ts[i] = scale(trigons[i], n);
	}
	const int flags = SDL_SWSURFACE;

	SDL_Surface* img = SDL_CreateRGBSurface(flags, xres * n,
		yres * n, 32, 0, 0, 0, 0);

	char filename[7];
	snprintf(filename, sizeof(filename), "%dX.bmp", n);

	draw(img, ts);
	SDL_SaveBMP(img, filename);
	SDL_FreeSurface(img);
}

void cleanAndQuit()
{
	SDL_SaveBMP(screen, "output.bmp");
	SDL_FreeSurface(screen);
	SDL_Quit();
	exit(0);
}

//----------------------------------------------------------------

int fully_mutate(SDL_Surface* s, uint32_t img[][yres], tri* trigons)
{
	int i = rand() % NUM_TRIGONS;

	tri t         = trigons[i];
	trigons[i]    = triangle();
	int64_t cost0 = cost(screen, img, t, trigons[i]);

	draw(s, trigons);

	if (cost0 < cost(screen, img, t, trigons[i]))
	{
		trigons[i] = t;
		return 0;
	}
	return 1;
}

int swap_adjacents(SDL_Surface* s, uint32_t img[][yres], tri* trigons)
{
	int i = rand() % NUM_TRIGONS;
	int j = i == 0 ? 1 : ((i == NUM_TRIGONS - 1) ? (i-1) : (i + SIGN(1)));

	int64_t cost0 = cost(s, img, trigons[i], trigons[j]);

	tri u = trigons[i];

	trigons[i] = trigons[j];
	trigons[j] = u;

	draw(s, trigons);

	if (cost0 < cost(s, img, trigons[i], u))
	{
		tri u = trigons[j];

		trigons[j] = trigons[i];
		trigons[i] = u;
		return 0;
	}
	return 1;
}

int arbitrary_swap(SDL_Surface* s, uint32_t img[][yres], tri* trigons)
{
	int i = rand() % NUM_TRIGONS;
	int j = rand() % NUM_TRIGONS;

	while (i == j)
	{
		i = rand() % NUM_TRIGONS;
		j = rand() % NUM_TRIGONS;
	}
	int64_t cost0 = cost(s, img, trigons[i], trigons[j]);

	tri u = trigons[i];

	trigons[i] = trigons[j];
	trigons[j] = u;

	draw(s, trigons);

	if (cost0 < cost(s, img, trigons[i], u))
	{
		tri u = trigons[j];

		trigons[j] = trigons[i];
		trigons[i] = u;
		return 0;
	}
	return 1;
}

int alter_alpha(SDL_Surface* s, uint32_t img[][yres], tri* trigons)
{
	int i = rand() % NUM_TRIGONS;
	tri t = trigons[i];
	tri u = copy_trigon(t);

	uint32_t a = u.p.a + SIGN(rand() % ALPHA_DELTA);

	u.p.a = a < MIN_ALPHA ? MIN_ALPHA : (a > MAX_ALPHA ? MAX_ALPHA : a);

	int64_t cost0 = tri_cost(s, img, t);
	trigons[i]    = u;

	draw(s, trigons);

	if (cost0 < tri_cost(s, img, u))
	{
		trigons[i] = t;
		return 0;
	}
	return 1;
}

int alter_channel(SDL_Surface* s, uint32_t img[][yres], tri* trigons)
{
	int i = rand() % NUM_TRIGONS;
	tri t = trigons[i];
	tri u = copy_trigon(t);

	switch (rand() % 3)
	{
		case 0:
			u.p.r = BYTE(u.p.r + SIGN(rand() % CHANNEL_DELTA));
			break;
		case 1:
			u.p.g = BYTE(u.p.g + SIGN(rand() % CHANNEL_DELTA));
			break;
		case 2:
			u.p.b = BYTE(u.p.b + SIGN(rand() % CHANNEL_DELTA));
	}
	int64_t cost0 = tri_cost(s, img, t);
	trigons[i]    = u;

	draw(s, trigons);

	if (cost0 < tri_cost(s, img, u))
	{
		trigons[i] = t;
		return 0;
	}
	return 1;
}

int alter_rgb(SDL_Surface* s, uint32_t img[][yres], tri* trigons)
{
	int i = rand() % NUM_TRIGONS;
	tri t = trigons[i];
	tri u = copy_trigon(t);

	int32_t r = u.p.r + SIGN(rand() % RGB_DELTA);
	int32_t g = u.p.g + SIGN(rand() % RGB_DELTA);
	int32_t b = u.p.b + SIGN(rand() % RGB_DELTA);

	u.p.r = r < 0 ? 0 : (r > 255 ? 255 : r);
	u.p.g = g < 0 ? 0 : (g > 255 ? 255 : g);
	u.p.b = b < 0 ? 0 : (b > 255 ? 255 : b);

	int64_t cost0 = tri_cost(s, img, t);
	trigons[i]    = u;

	draw(s, trigons);

	if (cost0 < tri_cost(s, img, u))
	{
		trigons[i] = t;
		return 0;
	}
	return 1;
}

//----------------------------------------------------------------

void trigonColr(char* s, tri t)
{
	char rgba[25];

	s[0] = '\0';

	sprintf(rgba, "%d", t.p.r);
	strcat(s, rgba);
	strcat(s, ",");
	sprintf(rgba, "%d", t.p.g);
	strcat(s, rgba);
	strcat(s, ",");
	sprintf(rgba, "%d", t.p.b);
	strcat(s, rgba);
	strcat(s, ",");
	snprintf(rgba, 9, "%f", t.p.a / 255.0);
	strcat(s, rgba);
}

void trigonXY(char* s, tri t)
{
	char n[4];

	s[0] = '\0';

	sprintf(n, "%d", t.a.x);
	strcat(s, n);
	strcat(s, ",");
	sprintf(n, "%d", t.a.y);
	strcat(s, n);
	strcat(s, " ");

	sprintf(n, "%d", t.b.x);
	strcat(s, n);
	strcat(s, ",");
	sprintf(n, "%d", t.b.y);
	strcat(s, n);
	strcat(s, " ");

	sprintf(n, "%d", t.c.x);
	strcat(s, n);
	strcat(s, ",");
	sprintf(n, "%d", t.c.y);
	strcat(s, n);
}

void write_SVG(tri* ts)
{
	const int SIZE = 36 + 120 * NUM_TRIGONS;

	char out[SIZE];  // should be big enough
	char s[30];

	FILE* fp = fopen("./output.svg", "w");
	out[0]   = '\0';
	s[0]     = '\0';

	strcat(out, "<svg height='");
	sprintf(s, "%d", yres);
	strcat(out, s);
	strcat(out, "' width='");
	s[0] = '\0';
	sprintf(s, "%d", xres);
	strcat(out, s);
	strcat(out, "'>\n");

	strcat(out, "<rect x='0' y='0' width='");
	s[0] = '\0';
	sprintf(s, "%d", xres);
	strcat(out, s);
	strcat(out, "' height='");
	s[0] = '\0';
	sprintf(s, "%d", yres);
	strcat(out, s);
	strcat(out, "' fill='rgba(");
	s[0] = '\0';
	sprintf(s, "%d", 0);  // FIXME
	strcat(out, s);
	strcat(out, ",");
	strcat(out, s);
	strcat(out, ",");
	strcat(out, s);
	strcat(out, ",");
	strcat(out, "1.0");
	strcat(out, ")' />");
	strcat(out, "\n");

	for (uint64_t i = 0; i < NUM_TRIGONS; i++)
	{
		strcat(out, "<polygon points='");
		trigonXY(s, ts[i]);
		strcat(out, s);
		strcat(out, "' fill='rgba(");
		trigonColr(s, ts[i]);
		strcat(out, s);
		strcat(out, ")' />");
		strcat(out, "\n");
	}
	strcat(out, "</svg>");

	if (fp != NULL)
	{
		fputs(out, fp);
		fclose(fp);
	}
}

void read_SVG(char* filename, tri* ts)
{
	FILE* fp = fopen(filename, "r");

	char str[5];
	int i = 0;

	while (i < NUM_TRIGONS)
	{
		point* a = &(ts[i]).a;
		point* b = &(ts[i]).b;
		point* c = &(ts[i]).c;
		rgba*  p = &(ts[i]).p;

		fgets(str, 5, fp);
		a->x = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		a->y = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		b->x = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		b->y = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		c->x = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		c->y = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		p->r = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		p->g = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		p->b = (uint32_t) strtol(str, NULL, 10);

		fgets(str, 5, fp);
		p->a = (uint32_t) strtol(str, NULL, 10);

		i++;
	}
	fclose(fp);
	exit(0);
}

//---------------------------------------------------------

int main(int argc, char *argv[])
{
	init(argc, argv[1]);

	uint32_t img[xres][yres];

	copyInputImage(img);
	sample_img(img);
	srand(time(NULL));

	tri trigons[NUM_TRIGONS];

	for (uint64_t i = 0; i < NUM_TRIGONS; i++)
	{
		trigons[i] = triangle();
	}

	if (argc > 2)
	{
		read_SVG(argv[1], trigons);
		printf("read");
	}

	int n = 0;        // number of improvements
	SDL_Event event;

	while (1)   // quit by pressing escape key
	{
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_KEYUP)
			{
				if (event.key.keysym.sym == SDLK_w)
				{
					printf("%d\n", n);
					write_SVG(trigons);
				}
				if (event.key.keysym.sym == SDLK_n)
				{
					printf("%d\n", n);
				}
				if (event.key.keysym.sym == SDLK_ESCAPE)
				{
					printf("%d\n", n);
					write_SVG(trigons);
					cleanAndQuit();
				}
				switch (event.key.keysym.sym)
				{
					case SDLK_1: writeScaledBMP(trigons, 1);  break;
					case SDLK_2: writeScaledBMP(trigons, 2);  break;
					case SDLK_3: writeScaledBMP(trigons, 3);  break;
					case SDLK_4: writeScaledBMP(trigons, 4);  break;
					case SDLK_5: writeScaledBMP(trigons, 5);  break;
					case SDLK_6: writeScaledBMP(trigons, 6);  break;
					case SDLK_7: writeScaledBMP(trigons, 7);  break;
					case SDLK_8: writeScaledBMP(trigons, 8);  break;
					case SDLK_9: writeScaledBMP(trigons, 9);  break;
					default: ;  // avoids 'unhandled case' warnings
				}
			}
		}
		draw(screen, trigons);

		if (rand() % 99859 == 0)  // fully randomize trigon
		{
			if (fully_mutate(screen, img, trigons))
			{
				printf("%d - MUT\n", n++);
				continue;
			}
		}
		if (rand() % 11 == 0)  // swap two arbitrary trigons
		{
			if (arbitrary_swap(screen, img, trigons))
			{
				printf("%d - SWP\n", n++);
				continue;
			}
		}
		if (rand() % 7 == 0)  // swap two adjacent trigons
		{
			if (swap_adjacents(screen, img, trigons))
			{
				printf("%d - ADJ\n", n++);
				continue;
			}
		}
		if (rand() % 5 == 0)  // alter trigon alpha
		{
			if (alter_alpha(screen, img, trigons))
			{
				printf("%d - AA\n", n++);
				continue;
			}
		}
		if (rand() % 3 == 0)  // alter rgb
		{
			if (alter_rgb(screen, img, trigons))
			{
				printf("%d - RGB\n", n++);
				continue;
			}
		}
		if (rand() % 2 == 0)  // alter 1 rgb channel
		{
			if (alter_channel(screen, img, trigons))
			{
				printf("%d - CHAN\n", n++);
				continue;
			}
		}
		SDL_Flip(screen);
		n++;
	}
}

