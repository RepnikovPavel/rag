# 1192 ms 
def iter_forward_gpu_buff(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        num_layers_in_buffer:int=None,
        verbose=False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        if num_layers_in_buffer % 2 != 0:
            raise ValueError("num_layers_in_buffer must be an even number.")
    
        chunk_size = num_layers_in_buffer // 2
        total_layers = self.config.num_hidden_layers
        layers = self.layers[:total_layers]
    
        hidden_states = inputs_embeds
        hidden_states_source_device=hidden_states.device
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        device_ = 'cuda'
        host_ = 'cpu'
        hidden_states=hidden_states.to(device_,non_blocking=True)
        device_position_embeddings=[el_.to(device_,non_blocking=True) for el_ in position_embeddings]
        device_position_ids=position_ids.to(device_,non_blocking=True)
        device_past_key_values=past_key_values.to(device_,non_blocking=True) if past_key_values is not None else None
        
        is_the_same_,attention_type_=validate_attention_types(self)
        if is_the_same_:
            if verbose:
                print(f'all attn type is {attention_type_}') 
        else:
            raise NotImplementedError

        device_attn_mask = causal_mask_mapping[attention_type_].to(device_,non_blocking=True)
        assert self.config.num_hidden_layers==36
        
        first_chunk = layers[0:chunk_size]
        move_layers_to_device_async(first_chunk, device_)
        # Переменная для отслеживания потока загрузки
        load_thread = None
        # Переменная для отслеживания потока выгрузки (опционально, для очистки памяти)
        offload_thread = None
        
        if verbose:
            pbar = tqdm(range(0, total_layers, chunk_size), desc='Double Buffered Forward Pass')
        else:
            pbar = range(0, total_layers, chunk_size)
        for i in pbar:
            current_chunk = layers[i : i + chunk_size]
            next_chunk_idx = i + chunk_size
            next_chunk = layers[next_chunk_idx : next_chunk_idx + chunk_size]

            # 1. Если запускали загрузку следующего чанка на прошлой итерации, ждем её завершения
            if load_thread is not None:
                load_thread.join()
            
            # Ждем выгрузки предыдущего чанка, чтобы освободить место (если VRAM критичен)
            if offload_thread is not None:
                offload_thread.join()

            # 2. Forward pass для текущего чанка
            # В это время GPU занят вычислениями
            for layer_idx, layer in enumerate(current_chunk):
                # Слои уже на GPU благодаря предзагрузке
                hidden_states = layer(
                    hidden_states,
                    attention_mask=device_attn_mask,
                    position_embeddings=device_position_embeddings,
                    position_ids=device_position_ids,
                    past_key_values=device_past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            # 3. Запускаем загрузку СЛЕДУЮЩЕГО чанка (асинхронно)
            # Это произойдет параллельно с тем, как мы будем выгружать текущие и готовиться к след циклу
            if next_chunk:
                load_thread = threading.Thread(
                    target=move_layers_to_device_async, 
                    args=(next_chunk, device_)
                )
                load_thread.start()
            else:
                load_thread = None

            # 4. Запускаем выгрузку ТЕКУЩЕГО чанка обратно в Host (асинхронно)
            # Это освобождает память для будущих чанков
            offload_thread = threading.Thread(
                target=move_layers_to_device_async, 
                args=(current_chunk, host_)
            )
            offload_thread.start()

        # Финальная синхронизация: ждем пока выгрузится последний чанк
        if offload_thread is not None:
            offload_thread.join()
        
        if load_thread is not None:
            load_thread.join()

        first_chunk = layers[0:chunk_size]
        move_layers_to_device_async(first_chunk, device_)
        
        hidden_states=hidden_states.to(hidden_states_source_device)
        hidden_states = self.norm(hidden_states)
        # BaseModelOutputWithPast:
        # last_hidden_state: torch.FloatTensor | None = None
        # past_key_values: Cache | None = None
        # hidden_states: tuple[torch.FloatTensor, ...] | None = None
        # attentions: tuple[torch.FloatTensor, ...] | None = None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )