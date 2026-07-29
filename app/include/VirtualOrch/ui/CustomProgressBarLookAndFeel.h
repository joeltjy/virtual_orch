#pragma once

#include <JuceHeader.h>

class CustomProgressBarLookAndFeel : public juce::LookAndFeel_V4 {
public:
    void drawProgressBar(Graphics &g, ProgressBar &progressBar, int width, int height, double progress,
                         const String &textToShow) override {
        auto background = progressBar.findColour(ProgressBar::backgroundColourId);
        auto foreground = progress >= 0 ? juce::Colours::darkolivegreen : juce::Colours::indianred;
        auto textColour = juce::Colours::whitesmoke;

        auto barBounds = progressBar.getLocalBounds().toFloat();

        g.setColour(background);
        g.fillRoundedRectangle(barBounds, (float) progressBar.getHeight() * 0.5f);

        auto textX = 0;
        auto textWidth = progressBar.getWidth();

        if (progress >= -1.0f && progress <= 1.0f) {
            Path p;
            p.addRoundedRectangle(barBounds, (float) progressBar.getHeight() * 0.5f);
            g.reduceClipRegion(p);

            if (progress >= 0) {
                barBounds.setX(barBounds.getWidth() / 2.f);
                textX = progressBar.getWidth() / 2;
                barBounds.setWidth(barBounds.getWidth() * ((float) progress / 2.f));
                textWidth = progressBar.getWidth() * ((float) progress / 2.f);
            } else {
                barBounds.setX((barBounds.getWidth() / 2.f) * (1 + (float) progress));
                textX = progressBar.getWidth() / 2 * (1 + (float) progress);
                barBounds.setWidth(barBounds.getWidth() * ((float) progress / -2.f));
                textWidth = progressBar.getWidth() * ((float) progress / -2.f);
            }
            g.setColour(foreground);
            g.fillRoundedRectangle(barBounds, (float) progressBar.getHeight() * 0.5f);

            g.setColour(textColour);
            g.setFont((float) height * 0.6f);

            g.drawText((progress >= 0 ? "+ " : "- ") + juce::String(abs(progress) * 10, 2) + "s", textX, 0,
                       textWidth, height, Justification::centred, false);
        } else {
            // spinning bar..
            g.setColour(background);

            auto stripeWidth = height * 2;
            auto position = static_cast<int>(Time::getMillisecondCounter() / 15) % stripeWidth;

            Path p;

            for (auto x = static_cast<float>(-position); x < (float) (width + stripeWidth); x += (float) stripeWidth)
                p.addQuadrilateral(x, 0.0f,
                                   x + (float) stripeWidth * 0.5f, 0.0f,
                                   x, static_cast<float>(height),
                                   x - (float) stripeWidth * 0.5f, static_cast<float>(height));

            Image im(Image::ARGB, width, height, true); {
                Graphics g2(im);
                g2.setColour(foreground);
                g2.fillRoundedRectangle(barBounds, (float) progressBar.getHeight() * 0.5f);
            }

            g.setTiledImageFill(im, 0, 0, 0.85f);
            g.fillPath(p);

            g.setColour(textColour);
            g.setFont((float) height * 0.6f);

            g.drawText(progress >= 0 ? "ALL GOOD" : "!! WARNING !!", 0, 0,
                       width,
                       height, Justification::centred, false);
        }
    }
};