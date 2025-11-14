import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { ChatComponent } from './chat/chat.component';
import { StoreModule } from '@ngrx/store';
import { EffectsModule } from '@ngrx/effects';
import { FormsModule } from '@angular/forms';
import { chatReducer } from './chat/store/chat.reducer';
import { ChatEffects } from './chat/store/chat.effects';
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  declarations: [AppComponent, ChatComponent],
  imports: [
    BrowserModule,
    StoreModule.forRoot({ chat: chatReducer }, {}),
    EffectsModule.forRoot([ChatEffects]),
    FormsModule,
    HttpClientModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
